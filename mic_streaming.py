import asyncio
import argparse
import sounddevice as sd
import numpy as np
from streaming.vad_streaming import VAD, _VADOptions, AudioFrame, VADStream
from streaming.onnx_model import new_inference_session
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--min_speech_duration",
        type=float,
        default=0.05,
        help="Minimum duration of speech to consider it as valid speech",
    )

    parser.add_argument(
        "--min_silence_duration",
        type=float,
        default=0.5,
        help="Minimum duration of silence to consider to do inference speech buffer",
    )

    parser.add_argument(
        "--activation_threshold",
        type=float,
        default=0.5,
        help="Threshold for speech activity detection",
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate of the audio input",
    )
    return parser.parse_args()

async def microphone_stream(vad_stream: VADStream, sample_rate: int = 16000, frame_duration: float = 0.1, duration: float = 10.0):
    samples_per_frame = int(sample_rate * frame_duration)
    
    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Stream status: {status}")
    
        if indata.shape[1] > 1:
            frame_data = indata[:, 0]
        else:
            frame_data = indata.flatten()
        
        frame_data = (frame_data * 32768).astype(np.int16)
        
        frame = AudioFrame(
            data=frame_data,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(frame_data)
        )
        
        asyncio.run_coroutine_threadsafe(vad_stream.push_frame(frame), asyncio.get_event_loop())
    
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', 
                           blocksize=samples_per_frame, callback=audio_callback):
            logger.info("Start recording from microphone...")
            await asyncio.sleep(duration)
    
    finally:
        await vad_stream.close()
        logger.info("Stop recording...")

async def receive_transcripts(vad_stream: VADStream):
    while not vad_stream.is_closed:
        transcript = await vad_stream.get_transcript()
        if transcript is None:
            if vad_stream.is_closed:
                break
            continue
        print(f"Received transcript: {transcript}")

async def main():
    args = get_args()

    opts = _VADOptions(
        min_speech_duration = args.min_speech_duration,
        min_silence_duration = args.min_silence_duration,
        activation_threshold = args.activation_threshold,
        sample_rate = args.sample_rate
    )
    
    session = new_inference_session(use_cpu=True)
    vad = VAD(session=session, opts=opts)
    vad_stream = vad.stream()
    
    record_duration = 10.0
    
    try:
        await asyncio.gather(
            microphone_stream(vad_stream, sample_rate=16000, frame_duration=0.1, duration=record_duration),
            receive_transcripts(vad_stream)
        )
    finally:
        await vad_stream.close()

if __name__ == "__main__":
    asyncio.run(main())