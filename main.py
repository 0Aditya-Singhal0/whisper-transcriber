import librosa
from utils import get_pipeline
from config import generation_args

# File path to your audio file
# audio_path = "path/to/your/audio/file"
audio_path = "/home/tagglabs/Desktop/code/assets/voices/sampled_aj_voice.mp3"

# process audio
audio, sampleRate = librosa.load(audio_path, sr=16000) # Whisper is trained to use 16kHz sampling rate

pipe = get_pipeline()


result = pipe(
    audio,
    return_timestamps=generation_args["return_timestamps"],
    # generate_kwargs=generation_args["generate_kwargs"],
)

print(result)