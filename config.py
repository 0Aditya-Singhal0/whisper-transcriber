import torch

"""
This file contains the config settings to load the whisper model in.
The config here follows default recommended settings for the model provided here https://huggingface.co/openai/whisper-large-v3

You can make your changes here to suit your needs. :D
"""

# Model params for loading the model
model_params = {
    # Change model you want to use here
    "model_id": "openai/whisper-large-v3",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "low_cpu_mem_usage": True,
    "use_safetensors": True,
    "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "eager",
}

# Pipeline params for using the pipeline
pipeline_params = {
    "max_new_tokens": 128,
    "chunk_length_s": 30,
    "batch_size": 16,
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
}

generation_args = {
    # Set to True if you want to return timestamps. 
    # Set to 'word' if you want word based stamps but disable attention implementation since WhisperFlashAttention2 attention does not support output_attentions
    "return_timestamps": False,
    
    #  Only use if you are working with multiple languages
    "generate_kwargs": {
        "language": "english",
        "task": "transcribe",
    },
}
