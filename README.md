# Voice message source identifier
Voice message source identifier is a tool for classifying the origin of voice messages based on AAC encoding and container features.
This project is associated with a research paper.

## An overview of the proposed methodology
![An overview of the proposed methodology](https://github.com/mjkim100ku/Voice-Message-Source-Identifier/blob/main/Proposed%20Methodology%20for%20Identifying%20the%20Source%20of%20Voice%20Messages.png)

## Dataset
You can download the dataset from the following [Google Drive link](https://drive.google.com/drive/folders/1dBcuxgXVlQmEppyMjYugHUMseZlXcF0S?usp=sharing).

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
3. Install FFmpeg
   ```bash   
   sudo apt update
   sudo apt install ffmpeg
   ```
5. Install other dependacies
   ```bash   
   pip install -r requirements.txt
   ```

## Usage
You can classify voice message sources by specifying the folder path containing audio files as an argument.
```bash
python classifyAudio_voting.py [path_to_folder_containing_audio_files]
```
