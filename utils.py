from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from config import model_params, pipeline_params


def get_model():
    """
    Returns:
        _type_: transformers.WhisperForConditionalGeneration
    """
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_params["model_id"],
        low_cpu_mem_usage=model_params["low_cpu_mem_usage"],
        use_safetensors=model_params["use_safetensors"],
        torch_dtype=model_params["torch_dtype"],
        device_map=model_params["device"],
        attn_implementation=model_params["attn_implementation"],
    )
    return model


def get_processor():
    """
    Returns:
        _type_: transformers.WhisperProcessor
    """
    processor = AutoProcessor.from_pretrained(model_params["model_id"])
    return processor


def get_pipeline():
    """
    Returns:
        _type_: transformers.AutomaticSpeechRecognitionPipeline
    """
    model = get_model()
    processor = get_processor()
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=pipeline_params["max_new_tokens"],
        chunk_length_s=pipeline_params["chunk_length_s"],
        batch_size=pipeline_params["batch_size"],
        torch_dtype=pipeline_params["torch_dtype"],
    )
    return pipe
