import asyncio
import argparse
import soundfile as sf
import numpy as np
from streaming.vad_streaming import VAD, _VADOptions, AudioFrame, VADStream
from streaming.onnx_model import new_inference_session

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to a single audio file to process",
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

async def audio_file_stream(vad_stream: VADStream, file_path: str, frame_duration: float = 0.1):
    with sf.SoundFile(file_path, 'r') as f:
        sample_rate = f.samplerate
        channels = f.channels

        data = f.read(dtype='int16')
        if channels > 1:
            data = np.mean(data, axis=1)
        
        samples_per_frame = int(sample_rate * frame_duration)
        total_samples = len(data)
        
        for i in range(0, total_samples, samples_per_frame):
            frame_data = data[i:i + samples_per_frame]

            frame = AudioFrame(
                data=frame_data,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(frame_data)
            )

            await vad_stream.push_frame(frame)
            
            # simulate real-time delay
            await asyncio.sleep(frame_duration)

    print("Finished processing audio file.")
    await vad_stream.close()

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
    
    session = new_inference_session(force_cpu=True)
    vad = VAD(session=session, opts=opts)
    vad_stream = vad.stream()
    file_path = args.audio_path
    try:
        await asyncio.gather(
            audio_file_stream(vad_stream, file_path, frame_duration=0.1),
            receive_transcripts(vad_stream)
        )
    finally:
        await vad_stream.close()

if __name__ == "__main__":
    asyncio.run(main())