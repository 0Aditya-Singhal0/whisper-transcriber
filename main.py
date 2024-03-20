import argparse
import librosa
from utils import get_pipeline
from config import generation_args

parser = argparse.ArgumentParser(description="Whisper Transcriber")
parser.add_argument("-a", "--audio-path", type=str, help="Path to the audio file")
args = parser.parse_args()

# Process audio
audio, sampleRate = librosa.load(
    args.audio_path, sr=16000
)  # Whisper is trained to use 16kHz sampling rate

pipe = get_pipeline()

result = pipe(
    audio,
    return_timestamps=generation_args["return_timestamps"],
    # generate_kwargs=generation_args["generate_kwargs"],
)

with open("output.txt", "w") as file:
    file.write(result["text"])
