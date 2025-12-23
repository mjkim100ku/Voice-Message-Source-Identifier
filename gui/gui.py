import os
import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QLabel, QTableWidgetItem

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cli.classify_audio_files import classify_audio_voting


def set_windows_appusermodelid(app_id: str):
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass


def get_app_icon(project_root: str) -> QIcon:
    ico = os.path.join(project_root, "gui", "logo.ico")
    png = os.path.join(project_root, "gui", "logo.png")

    if os.path.exists(ico):
        return QIcon(ico)
    if os.path.exists(png):
        return QIcon(png)
    return QIcon()


APP_LABEL_MAP = {
    "band": "Band",
    "messenger": "Messenger",
    "kakaotalk": "Kakaotalk",
    "line": "Line",
    "naverworks": "Naver Works",
    "session": "Session",
    "signal": "Signal",
    "skype": "Skype",
    "slack": "Slack",
    "viber": "Viber",
    "webex": "Webex",
    "wire": "Wire",
}


def normalize_app_name(raw: str):
    raw = (raw or "").strip().lower()

    if raw.startswith("and_") or raw.startswith("ios_"):
        suffix = raw.split("_", 1)[1] if "_" in raw else raw
    else:
        suffix = raw

    pretty = APP_LABEL_MAP.get(suffix, suffix.replace("_", " ").title())
    return suffix, pretty


class ClassifyWorker(QtCore.QObject):
    resultReady = QtCore.pyqtSignal(str, str, object)  # filename, raw_app, percent
    progress = QtCore.pyqtSignal(int, int)             # current, total
    finished = QtCore.pyqtSignal(bool)                 # cancelled?
    error = QtCore.pyqtSignal(str, str)                # filename, traceback

    def __init__(self, folder_path: str, exts=('.m4a', '.aac', '.mp4', '.3gp')):
        super().__init__()
        self.folder_path = folder_path
        self.exts = tuple(e.lower() for e in exts)
        self._abort = False

    def abort(self):
        self._abort = True

    @QtCore.pyqtSlot()
    def run(self):
        cancelled = False
        try:
            # Gather targets first so we can show determinate progress (and stable layout).
            targets = []
            for root, _, files in os.walk(self.folder_path):
                for f in files:
                    if f.lower().endswith(self.exts):
                        targets.append(os.path.join(root, f))

            total = len(targets)
            done = 0
            self.progress.emit(done, total)

            for full_path in targets:
                if self._abort:
                    cancelled = True
                    break

                try:
                    result = classify_audio_voting(full_path)
                    app, percent = self._normalize_result(result)
                    self.resultReady.emit(full_path, app, percent)
                except Exception:
                    self.error.emit(full_path, traceback.format_exc())

                done += 1
                self.progress.emit(done, total)

        finally:
            self.finished.emit(cancelled)

    @staticmethod
    def _normalize_result(result):
        # Supports either (app, percent) or just "app"
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            app = str(result[0])
            try:
                percent = float(result[1])
            except Exception:
                percent = None
            return app, percent
        return str(result), None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ui_path: str, app_icon: QIcon):
        super().__init__()

        uic.loadUi(ui_path, self)

        # Force icon (window + taskbar as much as possible)
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)

        self.selectButton = self.findChild(QtWidgets.QPushButton, 'pushButton_select')
        self.analyzeButton = self.findChild(QtWidgets.QPushButton, 'pushButton_analyze')
        self.folderLineEdit = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
        self.resultsTable = self.findChild(QtWidgets.QTableWidget, 'tableWidget')

        # Added in UI: always visible, only starts moving when analysis runs
        self.progressBar = self.findChild(QtWidgets.QProgressBar, 'progressBar')
        self.progressLabel = self.findChild(QtWidgets.QLabel, 'progressLabel')

        self._thread = None
        self._worker = None

        self._setup_table()
        self._setup_progress_idle()

        self.selectButton.clicked.connect(self.select)
        self.analyzeButton.clicked.connect(self.on_analyze_clicked)

    def _setup_table(self):
        t = self.resultsTable

        t.setColumnCount(6)
        t.setHorizontalHeaderLabels(['#', 'Filename', 'OS', 'Application', 'Icon', 'Probability'])

        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.setAlternatingRowColors(True)

        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(30)

        header = t.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)           # Filename widest
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)  # OS
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)  # Application
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Fixed)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)

        t.setColumnWidth(0, 44)
        t.setColumnWidth(4, 60)

        t.setSortingEnabled(True)
        t.horizontalHeader().sortIndicatorChanged.connect(self.renumber_rows)

    def _setup_progress_idle(self):
        if self.progressBar:
            self.progressBar.setVisible(True) 
            self.progressBar.setTextVisible(False)
            self.progressBar.setRange(0, 100)   
            self.progressBar.setValue(0)

        if self.progressLabel:
            self.progressLabel.setMinimumWidth(110)
            self.progressLabel.setMaximumWidth(110)
            self.progressLabel.setText("(   0 /    0)")

    def select(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folderLineEdit.setText(folder_path.replace('/', os.path.sep))

    def on_analyze_clicked(self):
        # Cancel if already running
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.abort()
            self.analyzeButton.setEnabled(False)  # avoid double clicks
            return

        folder_path = self.folderLineEdit.text().strip()
        if not folder_path or not os.path.isdir(folder_path):
            return

        # Reset UI
        self.resultsTable.setSortingEnabled(False)
        self.resultsTable.setRowCount(0)

        if self.progressBar:
            self.progressBar.setVisible(True)
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(0)
        if self.progressLabel:
            self.progressLabel.setText("(   0 /    0)")

        self.selectButton.setEnabled(False)
        self.analyzeButton.setText("Cancel")
        self.analyzeButton.setEnabled(True)

        # Thread + worker
        self._thread = QtCore.QThread(self)
        self._worker = ClassifyWorker(folder_path)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)

        self._worker.resultReady.connect(self.add_table_row)
        self._worker.progress.connect(self.on_progress)
        self._worker.error.connect(self.on_worker_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self.on_worker_finished)

        self._thread.start()

    def on_progress(self, current: int, total: int):
        if not self.progressBar or not self.progressLabel:
            return

        if total <= 0:
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(0)
            self.progressLabel.setText("(   0 /    0)")
            return

        if self.progressBar.maximum() != total:
            self.progressBar.setRange(0, total)

        self.progressBar.setValue(current)
        self.progressLabel.setText(f"({current:4d} / {total:4d})")

    def on_worker_error(self, filename: str, tb: str):
        pass

    def on_worker_finished(self):
        self.selectButton.setEnabled(True)
        self.analyzeButton.setEnabled(True)
        self.analyzeButton.setText("Analyze")

        self.resultsTable.setSortingEnabled(True)
        self.renumber_rows()

        self._thread = None
        self._worker = None

    def renumber_rows(self):
        t = self.resultsTable
        for r in range(t.rowCount()):
            item = t.item(r, 0)
            if item is None:
                item = QTableWidgetItem()
                t.setItem(r, 0, item)
            item.setText(str(r + 1))
            item.setData(Qt.EditRole, r + 1)
            item.setTextAlignment(Qt.AlignCenter)

    @QtCore.pyqtSlot(str, str, object)
    def add_table_row(self, filename: str, raw_app: str, percent):
        t = self.resultsTable
        row = t.rowCount()
        t.insertRow(row)

        # #
        num_item = QTableWidgetItem(str(row + 1))
        num_item.setTextAlignment(Qt.AlignCenter)
        num_item.setData(Qt.EditRole, row + 1)
        t.setItem(row, 0, num_item)

        # Filename
        t.setItem(row, 1, QTableWidgetItem(filename))

        # OS
        raw_lower = (raw_app or "").lower()
        if raw_lower.startswith("and_"):
            os_label = "Android"
        elif raw_lower.startswith("ios_"):
            os_label = "iOS"
        else:
            os_label = "-"
        os_item = QTableWidgetItem(os_label)
        os_item.setTextAlignment(Qt.AlignCenter)
        t.setItem(row, 2, os_item)

        # Application (remove and_/ios_ and prettify)
        suffix, pretty = normalize_app_name(raw_app)
        app_item = QTableWidgetItem(pretty)
        app_item.setTextAlignment(Qt.AlignCenter)
        t.setItem(row, 3, app_item)

        # Icon column: no text, only QLabel widget
        icon_item = QTableWidgetItem("")
        icon_item.setData(Qt.DisplayRole, "")
        t.setItem(row, 4, icon_item)

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "icon", f"{suffix}.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            label.setPixmap(pixmap.scaled(28, 28, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        t.setCellWidget(row, 4, label)

        # Percent (support 0~1 -> 0~100 display)
        if percent is None:
            pct_item = QTableWidgetItem("-")
            pct_item.setData(Qt.EditRole, -1.0)
        else:
            try:
                p = float(percent)
                if 0.0 <= p <= 1.0:
                    p *= 100.0
            except Exception:
                p = -1.0

            pct_item = QTableWidgetItem(f"{p:.1f}%")
            pct_item.setData(Qt.EditRole, p)

        pct_item.setTextAlignment(Qt.AlignCenter)
        t.setItem(row, 5, pct_item)


if __name__ == "__main__":
    # 2) Taskbar icon fix (Windows): AppUserModelID must be set before QApplication
    set_windows_appusermodelid("audio.source.identifier")

    app = QtWidgets.QApplication(sys.argv)

    icon = get_app_icon(PROJECT_ROOT)
    if not icon.isNull():
        app.setWindowIcon(icon)

    # UI path 
    ui_path = os.path.join(CURRENT_DIR, "main.ui")

    window = MainWindow(ui_path=ui_path, app_icon=icon)
    window.show()

    sys.exit(app.exec_())

