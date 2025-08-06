import ctranslate2
import librosa
import transformers
from huggingface_hub import snapshot_download
import os
import time
model_repo = "duonguyen/whisper-vietnamese-ct2"
model_dir = snapshot_download(repo_id=model_repo)

audio_path = "fill_in_audio_path_here"  # Replace with your audio file path
audio, _ = librosa.load(audio_path, sr=16000, mono=True)

processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-small", chunk_length=12)
start = time.time()
inputs = processor(audio, return_tensors="np", sampling_rate=16000, do_normalize=True)
features = ctranslate2.StorageView.from_array(inputs.input_features)

model = ctranslate2.models.Whisper(model_dir) 


language = "vi"
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        f"<|{language}|>",   # language code
        "<|transcribe|>",
        "<|notimestamps|>",
    ]
)

results = model.generate(features, [prompt])
transcription = processor.decode(results[0].sequences_ids[0], skip_special_tokens=True)

print("Transcription:", transcription)
print("Time taken:", time.time() - start, "seconds")