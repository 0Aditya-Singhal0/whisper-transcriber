# Whisper Transcriber

## Overview
This project utilizes OpenAI's Whisper model for audio transcription, providing an efficient and accurate method to convert speech in audio files to text. This project has been implemented by using https://huggingface.co/openai/whisper-large-v3

## Getting Started

### Prerequisites
Ensure you have Python installed on your system. This project has been set up and tested using the configurations and guidelines provided by the Hugging Face documentation and python 3.10.

### Installation
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using the following command:

```pip install -r requirements.txt```


**Note for macOS Users:** During installation, you may encounter issues with NVIDIA CUDA-related packages and `bitsandbytes`. These packages are not supported on macOS and should be commented out in the `requirements.txt` file before proceeding with the installation.

### Configuration
All configurations can be customized in the `config.py` file. This includes changing the model by updating the `model_id` in the `model_params` section.

### Running the Transcription
To transcribe an audio file, use the following command:

```python main.py --audio-path /path/to/file```

Ensure you replace `/path/to/file` with the actual path to the audio file you wish to transcribe.

## Support
For more detailed information about configurations and usage, refer to the [Hugging Face documentation](https://huggingface.co/docs).

## Contributing
We welcome contributions! Please feel free to submit pull requests or open issues to improve the project or fix problems.