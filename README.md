# Voice message source identifier
Voice message source identifier is a tool for classifying the origin of voice messages based on AAC encoding and container features.
This project is associated with a research paper.


## Installation
1. Clone the repository
   ```bash   
   git clone git@github.com:mjkim100ku/Voice-Message-Source-Identifier.git
   cd Voice-Message-Source-Identifier/
   ```
2. Create and activate the conda environment (<https://docs.rapids.ai/install/>)
    ```bash    
    conda create -n rapids-25.04 -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.04 python=3.12 'cuda-version>=12.0,<=12.8'
    conda activate rapids-25.04
    ```
3. Install other dependacies
   ```bash   
   pip install -r requirements.txt
   ```


## Getting Started
- Download `ffmpeg.exe` and place it under `tools/ffmpeg/ffmpeg.exe` when running from source.
- Download `ffprobe.exe` and place it under `tools/ffmpeg/ffprobe.exe` when running from source.
- Pre-built binaries are available in the repository releases.


## Dataset
You can download the dataset from the following [Google Drive link](https://drive.google.com/drive/folders/1dBcuxgXVlQmEppyMjYugHUMseZlXcF0S?usp=sharing).


## Usage (cli)
You can classify voice message sources by specifying the folder path containing audio files as an argument.
```bash
python classifyAudio_voting.py [path_to_folder_containing_audio_files]
```
