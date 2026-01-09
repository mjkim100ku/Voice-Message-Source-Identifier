import os
import tempfile
import pandas as pd
import binascii
import numpy as np
from datetime import datetime
from pymediainfo import MediaInfo
from collections import Counter, defaultdict
import json, sys, shutil
from pathlib import Path
from parsers.aac_parser import ParseADTS
from parsers.mp4_parser import MP4Parser
import subprocess

audio_object_type_map = {
    0:  "NULL",
    1:  "AAC MAIN",
    2:  "AAC LC",
    3:  "AAC SSR",
    4:  "AAC LTP",
    5:  "SBR",
    6:  "AAC scalable",
    7:  "TwinVQ",
    8:  "CELP",
    9:  "HVXC",
    10: "(reserved)",
    11: "(reserved)",
    12: "TTSI",
    13: "Main synthetic",
    14: "Wavetable synthesis",
    15: "General MIDI",
    16: "Algorithmic Synthesis and Audio FX",
    17: "ER AAC LC",
    18: "(reserved)",
    19: "ER AAC LTP",
    20: "ER AAC scalable",
    21: "ER TwinVQ",
    22: "ER BSAC",
    23: "ER AAC LD",
    24: "ER CELP",
    25: "ER HVXC",
    26: "ER HILN",
    27: "ER Parametric",
    28: "SSC",
    29: "(reserved)",
    30: "(reserved)",
    31: "(escape)",
    32: "Layer-1",
    33: "Layer-2",
    34: "Layer-3",
    35: "DST"
}

sampling_index_map = {
    0:  "96000",
    1:  "88200",
    2:  "64000",
    3:  "48000",
    4:  "44100",
    5:  "32000",
    6:  "24000",
    7:  "22050",
    8:  "16000",
    9:  "12000",
    10: "11025",
    11: "8000",
    12: "7350"
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)


def get_resource_path(*parts):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = PROJECT_ROOT
    return os.path.join(base, *parts)


def resolve_ffmpeg_executable(exe_name: str) -> str:
    env_path = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_DIR")
    if env_path:
        if os.path.isdir(env_path):
            candidate = os.path.join(env_path, exe_name)
        else:
            candidate = env_path
        if os.path.exists(candidate):
            return candidate

    candidates = [
        get_resource_path(exe_name),
        get_resource_path("ffmpeg", exe_name),
        get_resource_path("bin", exe_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return exe_name


def _subprocess_hide_window_kwargs():
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        return {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def get_bitrate_from_ffmpeg(file_path):
    try:
        ffprobe_exe = resolve_ffmpeg_executable("ffprobe.exe" if os.name == "nt" else "ffprobe")
        cmd = [
            ffprobe_exe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "json",
            file_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_subprocess_hide_window_kwargs(),
        )
        if result.returncode != 0:
            return "Unknown"
        probe = json.loads(result.stdout or "{}")
        streams = probe.get("streams", [])
        if streams:
            return streams[0].get("bit_rate", "Unknown")
        return "Unknown"
    except Exception as e:
        print(f"Error fetching bitrate from ffmpeg for {file_path}: {e}")
        return "Unknown"


def demux_to_adts(file_path, output_path):
    try:
        ffmpeg_exe = resolve_ffmpeg_executable("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            file_path,
            "-codec",
            "copy",
            "-f",
            "adts",
            output_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_subprocess_hide_window_kwargs(),
        )
        if result.returncode != 0:
            return False
    except Exception as e:
        print(f"Error during demuxing {file_path} to {output_path}: {e}")
        return False
    return True


def AacLcADTS(file_path):
    with open(file_path, 'rb') as f:
        buf = f.read()
    buf = binascii.hexlify(buf).decode('utf-8')
    buf = binascii.unhexlify(buf)

    all_frames = []
    all_huffmans = []
    all_sectlen = []
    offset = 0

    while offset < len(buf):
        # print(offset)
        adts, section_lengths, huffmancodebooks, err = ParseADTS(buf[offset:])
        
        if err is not None:
            print(f"Error parsing frame at offset {offset}: {err}")
            break
        
        def obj_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {key: obj_to_dict(value) for key, value in vars(obj).items()}
            elif isinstance(obj, list):
                return [obj_to_dict(item) for item in obj]
            else:
                return str(obj)

        adts_dict = obj_to_dict(adts)
        all_frames.append(adts_dict)

        huffman_dict = obj_to_dict(huffmancodebooks)
        all_huffmans.append(huffman_dict)

        sectlen_dict = obj_to_dict(section_lengths)
        all_sectlen.append(sectlen_dict)

        offset += adts.aac_frame_length

        if adts.aac_frame_length == 0:
            return all_frames, all_huffmans, all_sectlen

    return all_frames, all_huffmans, all_sectlen


def spectral_data(all_huffmans):
    flattened = [int(value) for sublist in all_huffmans for inner_list in sublist for value in inner_list]
    hcod_data = []
    for item in all_huffmans:
        for element in item:
            hcod_data.append([int(value) for value in element])

    huffman_indices = range(12)
    counts = [flattened.count(i) for i in huffman_indices]
    total_count = sum(counts)
    probabilities = [count / total_count for count in counts]
    cb_probabilities = {i: probabilities[i] for i in range(len(probabilities))}

    transitions_intra = defaultdict(int)
    total_transitions_intra = 0
    for row in hcod_data:
        for i in range(len(row) - 1):
            current_index = row[i]
            next_index = row[i + 1]
            transitions_intra[(current_index, next_index)] += 1
            total_transitions_intra += 1

    fa_cbmotp_intra = {key: cnt / total_transitions_intra for key, cnt in transitions_intra.items()}
    sorted_fa_cbmotp_intra = {k: fa_cbmotp_intra[k] for k in sorted(fa_cbmotp_intra.keys())}

    return cb_probabilities, sorted_fa_cbmotp_intra


def scalefactor_data(all_frames):
    dpcm_sf_data = []
    num_bands = 0

    for item in all_frames:
        if item["ChannelConfiguration"] == "1":
            for element in item.get("Single_channel_elements", []):
                ch_stream = element.get("Channel_stream", {})
                sf_data = ch_stream.get("Scale_factor_data", {})
                dpcm_sf_data.append(sf_data.get("dpcm_sf", []))
                ics_info = ch_stream.get("Ics_info", {})
                max_sfb_value = int(ics_info.get("Max_sfb", 0))
                if max_sfb_value > num_bands:
                    num_bands = max_sfb_value

        elif item["ChannelConfiguration"] == "2":
            for element in item.get("Channel_pair_elements", []):
                ch1 = element.get("Channel_stream1", {})
                ch2 = element.get("Channel_stream2", {})

                sf1 = ch1.get("Scale_factor_data", {})
                dpcm_sf_data.append(sf1.get("dpcm_sf", []))
                ics_info1 = ch1.get("Ics_info", {})
                max_sfb_val1 = int(ics_info1.get("Max_sfb", 0))
                if max_sfb_val1 > num_bands:
                    num_bands = max_sfb_val1

                sf2 = ch2.get("Scale_factor_data", {})
                dpcm_sf_data.append(sf2.get("dpcm_sf", []))
                ics_info2 = ch2.get("Ics_info", {})
                max_sfb_val2 = int(ics_info2.get("Max_sfb", 0))
                if max_sfb_val2 > num_bands:
                    num_bands = max_sfb_val2

    flattened = [int(value) for block in dpcm_sf_data for sublist in block for value in sublist]
    counts = Counter(flattened)
    total_count = sum(counts.values())
    dpcm_sf_probabilities = {k: counts.get(k, 0) / total_count for k in range(121)}
    sorted_dpcm_sf_prob = {k: dpcm_sf_probabilities[k] for k in sorted(dpcm_sf_probabilities.keys())}

    transitions_inter = defaultdict(int)
    total_transitions_inter = 0
    for i in range(len(dpcm_sf_data) - 1):
        current_block = dpcm_sf_data[i]
        next_block = dpcm_sf_data[i + 1]
        if len(current_block) > 1:
            for j in range(len(current_block) - 1):
                cur_sub = current_block[j]
                nxt_sub = current_block[j + 1]
                if cur_sub and nxt_sub:
                    c_idx = int(cur_sub[-1])
                    n_idx = int(nxt_sub[0])
                    transitions_inter[(c_idx, n_idx)] += 1
                    total_transitions_inter += 1
        if current_block and next_block:
            c_idx = int(current_block[-1][-1]) if current_block[-1] else None
            n_idx = int(next_block[0][0]) if next_block[0] else None
            if c_idx is not None and n_idx is not None:
                transitions_inter[(c_idx, n_idx)] += 1
                total_transitions_inter += 1

    fa_sfmotp_inter = {k: v / total_transitions_inter for k, v in transitions_inter.items()}

    transitions_intra = defaultdict(int)
    total_transitions_intra = 0
    for block in dpcm_sf_data:
        for sublist in block:
            for j in range(len(sublist) - 1):
                c_val = int(sublist[j])
                n_val = int(sublist[j + 1])
                transitions_intra[(c_val, n_val)] += 1
                total_transitions_intra += 1

    fa_sfmotp_intra = {k: v / total_transitions_intra for k, v in transitions_intra.items()}

    sorted_fa_sfmotp_inter = {k: fa_sfmotp_inter[k] for k in sorted(fa_sfmotp_inter.keys())}
    sorted_fa_sfmotp_intra = {k: fa_sfmotp_intra[k] for k in sorted(fa_sfmotp_intra.keys())}

    return sorted_dpcm_sf_prob, sorted_fa_sfmotp_inter, sorted_fa_sfmotp_intra


def section_data(all_frames, all_sectlen):
    num_sec_data = []
    for item in all_frames:
        if item["ChannelConfiguration"] == "1":
            for element in item.get("Single_channel_elements", []):
                ch_stream = element.get("Channel_stream", {})
                sec_data = ch_stream.get("Section_data", {})
                num_sec_data.extend([int(v) for v in sec_data.get("num_sec", [])])
        elif item["ChannelConfiguration"] == "2":
            for element in item.get("Channel_pair_elements", []):
                ch1 = element.get("Channel_stream1", {})
                ch2 = element.get("Channel_stream2", {})
                sec1 = ch1.get("Section_data", {})
                sec2 = ch2.get("Section_data", {})
                num_sec_data.extend([int(v) for v in sec1.get("num_sec", [])])
                num_sec_data.extend([int(v) for v in sec2.get("num_sec", [])])

    flattened_sect_len_data = [
        int(v)
        for block in all_sectlen
        for sublist in block
        for inner_list in sublist
        for v in inner_list
    ]

    num_sec_counts = Counter(num_sec_data)
    total_num_sec = sum(num_sec_counts.values())
    num_sec_prob = {k: v / total_num_sec for k, v in num_sec_counts.items()}

    sect_len_counts = Counter(flattened_sect_len_data)
    total_sect_len = sum(sect_len_counts.values())
    sect_len_prob = {k: v / total_sect_len for k, v in sect_len_counts.items()}

    transitions = defaultdict(int)
    total_transitions = 0
    for block in all_sectlen:
        for sublist in block:
            merged_list = [int(x) for inner_list in sublist for x in inner_list]
            for i in range(len(merged_list) - 1):
                c_val = merged_list[i]
                n_val = merged_list[i + 1]
                transitions[(c_val, n_val)] += 1
                total_transitions += 1
    sect_len_motp_ = {k: v / total_transitions for k, v in transitions.items()}

    sorted_num_sec_prob = {k: num_sec_prob[k] for k in sorted(num_sec_prob.keys())}
    sorted_sect_len_prob = {k: sect_len_prob[k] for k in sorted(sect_len_prob.keys())}
    sorted_sect_len_motp = {k: sect_len_motp_[k] for k in sorted(sect_len_motp_.keys())}

    return sorted_num_sec_prob, sorted_sect_len_prob, sorted_sect_len_motp


def flatten_iso_boxes(data, prefix="", array_keys_to_flatten=None, excluded_paths=None):
    if array_keys_to_flatten is None:
        array_keys_to_flatten = {"entries", "extensions", "descriptors", "sub_descriptors", "ilst"}
    if excluded_paths is None:
        excluded_paths = {
            "moov/trak/mdia/minf/stbl/stsz/entries",
            "moov/trak/mdia/minf/stbl/stco/entries"
        }

    items = {}

    if isinstance(data, dict):
        for k, v in data.items():
            if k == "mdat":
                continue
            new_prefix = f"{prefix}/{k}" if prefix else k

            if isinstance(v, dict):
                items.update(flatten_iso_boxes(v, prefix=new_prefix,
                                          array_keys_to_flatten=array_keys_to_flatten,
                                          excluded_paths=excluded_paths))
            elif isinstance(v, list):
                if new_prefix == "moov/trak/mdia/minf/stbl/stts/entries":
                    for i, elem in enumerate(v):
                        if i == 0:
                            items.update(flatten_iso_boxes(elem,
                                                      prefix=f"{new_prefix}[0]",
                                                      array_keys_to_flatten=array_keys_to_flatten,
                                                      excluded_paths=excluded_paths))
                        else:
                            base_prefix = new_prefix.rsplit('/', 1)[0]
                            col_name = f"{base_prefix}/@entries[{i}]"
                            items[col_name] = f"[{elem}]"
                    continue

                if k in array_keys_to_flatten:
                    if new_prefix in excluded_paths:
                        col_name = f"{prefix}/@{k}" if prefix else f"@{k}"
                        items[col_name] = v
                    else:
                        for i, elem in enumerate(v):
                            indexed_prefix = f"{new_prefix}[{i}]"
                            if isinstance(elem, (dict, list)):
                                items.update(flatten_iso_boxes(elem,
                                                          prefix=indexed_prefix,
                                                          array_keys_to_flatten=array_keys_to_flatten,
                                                          excluded_paths=excluded_paths))
                            else:
                                col_name = f"{indexed_prefix}/@value"
                                items[col_name] = elem
                else:
                    col_name = f"{prefix}/@{k}" if prefix else f"@{k}"
                    items[col_name] = v

            else:
                col_name = f"{prefix}/@{k}" if prefix else f"@{k}"
                items[col_name] = v

    elif isinstance(data, list):
        col_name = prefix if prefix else "@root_array"
        items[col_name] = data
    else:
        col_name = prefix if prefix else "@root_scalar"
        items[col_name] = data

    return items


def extract_audio_info(path):
    audio_info = []

    excluded_flattened_keys = {
        "moov/trak/mdia/minf/stbl/stsd/entries[0]/extensions[0]/descriptors[0]/sub_descriptors[1]/audio_specific_config/@audio_object_type",
        "moov/trak/mdia/minf/stbl/stsd/entries[0]/extensions[0]/descriptors[0]/sub_descriptors[1]/audio_specific_config/@sampling_frequency_index",
        "moov/trak/mdia/minf/stbl/stsd/entries[0]/@channel_count",
        "moov/trak/mdia/minf/stbl/stsd/entries[0]/extensions[0]/descriptors[0]/sub_descriptors[0]/@avg_bitrate",
        "moov/trak/mdia/minf/stbl/stts/entries[0]/@sample_delta",
    }

    def process_file(file_path, file_name):
        temp_adts_path = os.path.join(
            tempfile.gettempdir(),
            os.path.basename(os.path.splitext(file_path)[0]) + "_temp.aac",
        )
        if os.path.exists(temp_adts_path):
            os.remove(temp_adts_path)
            
        # print(f"Processing: {file_path}")

        try:
            media_info = MediaInfo.parse(file_path)
            container_mediainfo = "Unknown"
            bitrate_mode = "Unknown"

            if media_info.general_tracks:
                container_mediainfo = getattr(media_info.general_tracks[0], "format", "Unknown")

            for track in media_info.tracks:
                if track.track_type == "Audio":
                    bitrate_mode = getattr(track, "bit_rate_mode", "Unknown")
                    break

            row_info = {
                "file": file_name,
                "codec": "Unknown",
                "sampling_rate": "Unknown",
                "samples_per_frame": "Unknown",
                "framerate": "Unknown",
                "channels": "Unknown",
                "bitrate": "Unknown",
                "bitrate_mode": bitrate_mode,
                "cb": "Unknown",
                "cbMOTPintra": "Unknown",
                "dpcm_sf_probabilities": "Unknown",
                "fa_sfmotp_inter": "Unknown",
                "fa_sfmotp_intra": "Unknown",
                "num_sec_probabilities": "Unknown",
                "sect_len_probabilities": "Unknown",
                "sect_len_motp": "Unknown",
            }

            if container_mediainfo.upper() == "ADTS":
                audio_bitrate = get_bitrate_from_ffmpeg(file_path)
                row_info["bitrate"] = audio_bitrate

                frames, huffmans, sectlen = AacLcADTS(file_path)
                row_info["codec"] = audio_object_type_map.get(int(frames[0]["Profile"]), "Unknown")
                row_info["sampling_rate"] = frames[0]["SamplingFrequency"]
                row_info["samples_per_frame"] = frames[0]["Frame_length"]
                row_info["channels"] = frames[0]["ChannelConfiguration"]
                try:
                    sr_val = float(row_info["sampling_rate"])
                    spf_val = float(row_info["samples_per_frame"])
                    if spf_val != 0:
                        framerate = sr_val / spf_val
                        row_info["framerate"] = f"{framerate:.3f}"
                    else:
                        row_info["framerate"] = "Unknown"
                except:
                    pass

                cb, cbMOTPintra = spectral_data(huffmans)
                dpcm_sf, fa_sfmotp_inter, fa_sfmotp_intra = scalefactor_data(frames)
                num_sec, sect_len_prob, sect_len_motp_ = section_data(frames, sectlen)

                row_info["cb"] = cb
                row_info["cbMOTPintra"] = cbMOTPintra
                row_info["dpcm_sf_probabilities"] = dpcm_sf
                row_info["fa_sfmotp_inter"] = fa_sfmotp_inter
                row_info["fa_sfmotp_intra"] = fa_sfmotp_intra
                row_info["num_sec_probabilities"] = num_sec
                row_info["sect_len_probabilities"] = sect_len_prob
                row_info["sect_len_motp"] = sect_len_motp_

                audio_info.append(row_info)
                return

            elif container_mediainfo.upper() in ["MPEG-4", "MP4"]:
                MP4 = MP4Parser(file_path)
                MP4.parse()
                flattened = flatten_iso_boxes(MP4.atoms)

                aot_key = (
                    "moov/trak/mdia/minf/stbl/stsd/entries[0]/extensions[0]/descriptors[0]"
                    "/sub_descriptors[1]/audio_specific_config/@audio_object_type"
                )
                if aot_key in flattened:
                    val = int(flattened[aot_key])
                    row_info["codec"] = audio_object_type_map.get(val, "Unknown")

                sfi_key = (
                    "moov/trak/mdia/minf/stbl/stsd/entries[0]/extensions[0]/descriptors[0]"
                    "/sub_descriptors[1]/audio_specific_config/@sampling_frequency_index"
                )
                if sfi_key in flattened:
                    val = int(flattened[sfi_key])
                    row_info["sampling_rate"] = sampling_index_map.get(val, "Unknown")

                ch_key = "moov/trak/mdia/minf/stbl/stsd/entries[0]/extensions[0]/descriptors[0]/sub_descriptors[1]/audio_specific_config/@channel_configuration"
                if ch_key in flattened:
                    row_info["channels"] = flattened[ch_key]

                br_key = (
                    "moov/trak/mdia/minf/stbl/stsd/entries[0]/extensions[0]/descriptors[0]"
                    "/sub_descriptors[0]/@avg_bitrate"
                )
                if br_key in flattened:
                    row_info["bitrate"] = flattened[br_key]

                sd_key = "moov/trak/mdia/minf/stbl/stts/entries[0]/@sample_delta"
                if sd_key in flattened:
                    row_info["samples_per_frame"] = flattened[sd_key]

                try:
                    sr_str = row_info["sampling_rate"]
                    spf_str = row_info["samples_per_frame"]
                    sr_val = float(sr_str)
                    spf_val = float(spf_str)
                    if spf_val != 0:
                        framerate = sr_val / spf_val
                        row_info["framerate"] = f"{framerate:.3f}"
                    else:
                        row_info["framerate"] = "Unknown"
                except:
                    pass

                for ex_key in excluded_flattened_keys:
                    if ex_key in flattened:
                        del flattened[ex_key]

                for k, v in flattened.items():
                    row_info[k] = v

                if demux_to_adts(file_path, temp_adts_path):
                    frames, huffmans, sectlen = AacLcADTS(temp_adts_path)
                    cb, cbMOTPintra = spectral_data(huffmans)
                    dpcm_sf, fa_sfmotp_inter, fa_sfmotp_intra = scalefactor_data(frames)
                    num_sec, sect_len_prob, sect_len_motp_ = section_data(frames, sectlen)

                    row_info["cb"] = cb
                    row_info["cbMOTPintra"] = cbMOTPintra
                    row_info["dpcm_sf_probabilities"] = dpcm_sf
                    row_info["fa_sfmotp_inter"] = fa_sfmotp_inter
                    row_info["fa_sfmotp_intra"] = fa_sfmotp_intra
                    row_info["num_sec_probabilities"] = num_sec
                    row_info["sect_len_probabilities"] = sect_len_prob
                    row_info["sect_len_motp"] = sect_len_motp_

                    if os.path.exists(temp_adts_path):
                        os.remove(temp_adts_path)

                audio_info.append(row_info)

            else:
                audio_info.append(row_info)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                process_file(file_path, file)
    else:
        file_path = path
        file_name = os.path.basename(path)
        process_file(file_path, file_name)

    return audio_info


def run_extract(file_path):
    audio_dataset_info = extract_audio_info(file_path)

    df = pd.DataFrame(audio_dataset_info)

    preferred_right_columns = [
        "cb",
        "cbMOTPintra",
        "dpcm_sf_probabilities",
        "fa_sfmotp_inter",
        "fa_sfmotp_intra",
        "num_sec_probabilities",
        "sect_len_probabilities",
        "sect_len_motp"
    ]

    all_cols = list(df.columns)
    front_cols = [c for c in all_cols if c not in preferred_right_columns]
    final_cols = front_cols + preferred_right_columns
    df = df[final_cols]

    output_path = '.tmp'
    path = Path(output_path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = f"{output_path + os.path.sep}audio_dataset_info_{current_time}.csv"
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    # print(f"Audio dataset info saved to {output_csv_path}")

    return output_csv_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_folder = sys.argv[1]
    else:
        dataset_folder = "01dataset"

    run_extract(dataset_folder)
