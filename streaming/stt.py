import ctranslate2
import transformers
from huggingface_hub import snapshot_download

import asyncio
import numpy as np
import os
from scipy.signal import resample
import soundfile as sf
import nest_asyncio
from time import time
# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

import logging
logger = logging.getLogger("whisper STT logging")

class STT:
    def __init__(
        self,
        model_name: str = "duonguyen/whisper-vietnamese-ct2",
    ):
        model_dir = snapshot_download(repo_id=model_name)
        self._model = ctranslate2.models.Whisper(model_dir, compute_type="int8")
        self._processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-small", chunk_length=12)

    async def recognize(
        self,
        buffer,
    ):
        try:
            start_time = time()
            data, samplerate = buffer
            wav = data.astype(np.float32) / np.iinfo(np.int16).max
            if samplerate != 16000:
                num_samples = int(len(wav) * 16000 / samplerate)
                wav = resample(wav, num_samples)

            inputs = self._processor(wav, return_tensors="np", sampling_rate=16000, do_normalize=True)
            features = ctranslate2.StorageView.from_array(inputs.input_features)
            prompt_tokens = self._processor.tokenizer.convert_tokens_to_ids([
                "<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"
            ])
            results = self._model.generate(features, [prompt_tokens])
            transcript = self._processor.decode(results[0].sequences_ids[0], skip_special_tokens=True)
            respond_time = time()-start_time
            print(f"Transcription time: {respond_time:.2f} seconds")
            return transcript

        except Exception as e:
            logger.debug(e)