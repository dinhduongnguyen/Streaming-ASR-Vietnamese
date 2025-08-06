
from __future__ import annotations

import asyncio
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal
import numpy as np
import onnxruntime  # type: ignore
import soundfile as sf
from scipy import signal
from fractions import Fraction
import os

from .vad_filter import ExpFilter
from .onnx_model import OnnxModel
from .stt import STT

import logging
logger = logging.getLogger("whisper STT logging")

SLOW_INFERENCE_THRESHOLD = 0.2  # late by 200ms

stt_client = STT(model_name='duonguyen/whisper-vietnamese-ct2')

@dataclass
class _VADOptions:
    min_speech_duration: float = 0.05
    min_silence_duration: float = 0.4
    prefix_padding_duration: float = 0.2
    max_buffered_speech: float = 12.0
    activation_threshold: float = 0.5
    sample_rate: int = 16000

@dataclass
class AudioFrame:
    data: np.ndarray
    sample_rate: int
    num_channels: int
    samples_per_channel: int

    @classmethod
    def from_numpy(cls, data: np.ndarray, sample_rate: int, num_channels: int = 1):
        return cls(
            data=data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=len(data) // num_channels
        )

def combine_frames(frames: list[AudioFrame]) -> AudioFrame:
    if not frames:
        raise ValueError("No frames to combine")
    
    # Lọc các frame có dữ liệu hợp lệ (không rỗng)
    valid_frames = [frame for frame in frames if frame.data.size > 0]
    if not valid_frames:
        raise ValueError("No valid frames with non-empty data to combine")
    
    sample_rate = valid_frames[0].sample_rate
    num_channels = valid_frames[0].num_channels
    # Đảm bảo frame.data là mảng 1D
    combined_data = np.concatenate([frame.data.flatten() for frame in valid_frames])
    
    return AudioFrame(
        data=combined_data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=len(combined_data) // num_channels
    )

class AudioResampler:
    def __init__(self, input_rate: int, output_rate: int):
        self.input_rate = input_rate
        self.output_rate = output_rate
        frac = Fraction(output_rate, input_rate).limit_denominator()
        self.up = frac.numerator
        self.down = frac.denominator

    def push(self, frame: AudioFrame) -> list[AudioFrame]:
        resampled_data = signal.resample_poly(frame.data, self.up, self.down)
        return [AudioFrame(
            data=resampled_data,
            sample_rate=self.output_rate,
            num_channels=frame.num_channels,
            samples_per_channel=len(resampled_data) // frame.num_channels
        )]

class VAD:
    """
    Silero Voice Activity Detection (VAD) class.

    This class provides functionality to detect speech segments within audio data using the Silero VAD model.
    """  # noqa: E501

    def __init__(
        self,
        *,
        session: onnxruntime.InferenceSession,
        opts: _VADOptions,
    ) -> None:
        self._onnx_session = session
        self._opts = opts
        self._streams = weakref.WeakSet[VADStream]()

    def stream(self) -> VADStream:
        """
        Create a new VADStream for processing audio data.

        Returns:
            VADStream: A stream object for processing audio input and detecting speech.
        """
        stream = VADStream(
            self,
            self._opts,
            OnnxModel(
                onnx_session=self._onnx_session, sample_rate=self._opts.sample_rate
            ),
        )
        self._streams.add(stream)
        return stream


class VADStream:
    def __init__(self, vad: VAD, opts: _VADOptions, model: OnnxModel) -> None:
        self._task = asyncio.create_task(self._main_task())
        self._opts, self._model = opts, model
        self._loop = asyncio.get_event_loop()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._task.add_done_callback(lambda _: self._executor.shutdown(wait=True))
        self._exp_filter = ExpFilter(alpha=0.35)

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0  # (input_sample_rate)
        self._input_queue = asyncio.Queue()
        self._transcript_queue = asyncio.Queue()
        self._is_closed = False

    async def push_frame(self, frame: AudioFrame) -> None:
        await self._input_queue.put(frame)

    async def close(self) -> None:
        await self._input_queue.put(None)

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    async def get_transcript(self) -> str | None:
        try:
            return await asyncio.wait_for(self._transcript_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None


    async def _main_task(self) -> None:
        inference_f32_data = np.empty(self._model.window_size_samples, dtype=np.float32)
        speech_buffer_index: int = 0

        # "pub_" means public, these values are exposed to the users through events
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0
        infer_speech_duration = 0.0

        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0

        input_frames: list[AudioFrame] = []
        inference_frames: list[AudioFrame] = []
        resampler: AudioResampler | None = None
        temp_transcript = ''

        # used to avoid drift when the sample_rate ratio is not an integer
        input_copy_remaining_fract = 0.0

        extra_inference_time = 0.0

        try:
            while True:

                input_frame = await self._input_queue.get()

                if input_frame is None:
                    self._is_closed = True
                    await self._transcript_queue.put(None)
                    break

                if not self._input_sample_rate:
                    self._input_sample_rate = input_frame.sample_rate

                    # alloc the buffers now that we know the input sample rate
                    self._prefix_padding_samples = int(
                        self._opts.prefix_padding_duration * self._input_sample_rate
                    )

                    self._speech_buffer = np.empty(
                        int(self._opts.max_buffered_speech * self._input_sample_rate)
                        + self._prefix_padding_samples,
                        dtype=np.int16,
                    )

                    if self._input_sample_rate != self._opts.sample_rate:
                        # resampling needed: the input sample rate isn't the same as the model's
                        # sample rate used for inference
                        resampler = AudioResampler(
                            input_rate=self._input_sample_rate,
                            output_rate=self._opts.sample_rate,
                        )

                elif self._input_sample_rate != input_frame.sample_rate:
                    logger.error("a frame with another sample rate was already pushed")
                    continue

                assert self._speech_buffer is not None

                if input_frame.data.size > 0:
                    input_frames.append(input_frame)
                    if resampler is not None:
                        resampled_frames = resampler.push(input_frame)
                        inference_frames.extend([f for f in resampled_frames if f.data.size > 0])
                    else:
                        inference_frames.append(input_frame)

                while True:
                    start_time = time.perf_counter()

                    available_inference_samples = sum(
                        [frame.samples_per_channel for frame in inference_frames]
                    )
                    if available_inference_samples < self._model.window_size_samples:
                        break  # not enough samples to run inference

                    valid_inference_frames = [f for f in inference_frames if f.data.size > 0]
                    if not valid_inference_frames:
                        break

                    try:
                        input_frame = combine_frames([f for f in input_frames if f.data.size > 0])
                        inference_frame = combine_frames(valid_inference_frames)
                    except ValueError as e:
                        logger.warning(f"Failed to combine frames: {e}")
                        input_frames = []
                        inference_frames = []
                        break

                    # convert data to f32
                    np.divide(
                        inference_frame.data[: self._model.window_size_samples],
                        np.iinfo(np.int16).max,
                        out=inference_f32_data,
                        dtype=np.float32,
                    )

                    # run the inference
                    p = await self._loop.run_in_executor(
                        self._executor, self._model, inference_f32_data
                    )
                    p = self._exp_filter.apply(exp=1.0, sample=p)
                    #print(f"result vad: {p}")
                    #logger.error(f"success infer vad: {p}")
                    window_duration = self._model.window_size_samples / self._opts.sample_rate

                    pub_current_sample += self._model.window_size_samples
                    pub_timestamp += window_duration

                    resampling_ratio = self._input_sample_rate / self._model.sample_rate
                    to_copy = (
                        self._model.window_size_samples * resampling_ratio + input_copy_remaining_fract
                    )
                    to_copy_int = int(to_copy)
                    input_copy_remaining_fract = to_copy - to_copy_int

                    # copy the inference window to the speech buffer
                    available_space = len(self._speech_buffer) - speech_buffer_index
                    to_copy_buffer = min(to_copy_int, available_space)
                    if to_copy_buffer > 0:
                        self._speech_buffer[
                            speech_buffer_index : speech_buffer_index + to_copy_buffer
                        ] = input_frame.data[:to_copy_buffer]
                        speech_buffer_index += to_copy_buffer
                    elif not self._speech_buffer_max_reached:
                        # reached self._opts.max_buffered_speech (padding is included)
                        speech_buffer_max_reached = True
                        logger.warning(
                            "max_buffered_speech reached, ignoring further data for the current speech input"  # noqa: E501
                        )

                    inference_duration = time.perf_counter() - start_time
                    extra_inference_time = max(
                        0.0,
                        extra_inference_time + inference_duration - window_duration,
                    )
                    if inference_duration > SLOW_INFERENCE_THRESHOLD:
                        logger.warning(
                            "inference is slower than realtime",
                            extra={"delay": extra_inference_time},
                        )

                    def _reset_write_cursor() -> None:
                        nonlocal speech_buffer_index, speech_buffer_max_reached
                        assert self._speech_buffer is not None

                        if speech_buffer_index <= self._prefix_padding_samples:
                            return

                        padding_data = self._speech_buffer[
                            speech_buffer_index - self._prefix_padding_samples : speech_buffer_index
                        ]

                        self._speech_buffer_max_reached = False
                        self._speech_buffer[: self._prefix_padding_samples] = padding_data
                        speech_buffer_index = self._prefix_padding_samples

                    def _copy_speech_buffer() -> Tuple[np.ndarray, int]:
                        assert self._speech_buffer is not None
                        return self._speech_buffer[:speech_buffer_index], self._input_sample_rate

                    if pub_speaking:
                        pub_speech_duration += window_duration
                    else:
                        pub_silence_duration += window_duration


                    if p >= self._opts.activation_threshold:
                        speech_threshold_duration += window_duration
                        silence_threshold_duration = 0.0

                        if not pub_speaking:
                            if speech_threshold_duration >= self._opts.min_speech_duration:
                                pub_speaking = True
                                pub_silence_duration = 0.0
                                pub_speech_duration = speech_threshold_duration
                        
                        if (
                            pub_speaking
                            and pub_speech_duration >= (infer_speech_duration + 1)
                        ):
                            try:
                                temp_transcript = await stt_client.recognize(_copy_speech_buffer())
                                trans_data = {'transcript': temp_transcript, 'status': 'temporary'}
                                await self._transcript_queue.put(trans_data)
                                infer_speech_duration = pub_speech_duration
                            except Exception as e:
                                logger.error(f"ASR failed for transcription: {e}")

                    else:
                        silence_threshold_duration += window_duration
                        speech_threshold_duration = 0.0

                        if not pub_speaking:
                            _reset_write_cursor()

                        if (
                            pub_speaking
                            and silence_threshold_duration >= self._opts.min_silence_duration
                        ):
                            if speech_buffer_index > self._prefix_padding_samples:
                                try:
                                    if pub_speech_duration - infer_speech_duration >= 0.16:
                                        transcript = await stt_client.recognize(_copy_speech_buffer())
                                        trans_data = {'transcript': transcript, 'status': 'final'}
                                        await self._transcript_queue.put(trans_data)
                                    else:
                                        transcript = temp_transcript
                                        trans_data = {'transcript': transcript, 'status': 'final'}
                                        await self._transcript_queue.put(trans_data)

                                except Exception as e:
                                    logger.error(f"ASR failed for final transcription: {e}")

                            pub_speaking = False
                            pub_speech_duration = 0.0
                            infer_speech_duration = 0.0
                            pub_silence_duration = silence_threshold_duration

                            _reset_write_cursor()

                    # remove the frames that were used for inference from the input and inference frames
                    input_frames = []
                    inference_frames = []

                    # add the remaining data
                    if len(input_frame.data) - to_copy_int > 0:
                        data = input_frame.data[to_copy_int:]
                        input_frames.append(
                            AudioFrame(
                                data=data,
                                sample_rate=self._input_sample_rate,
                                num_channels=1,
                                samples_per_channel=len(data) // 2,
                            )
                        )

                    if len(inference_frame.data) - self._model.window_size_samples > 0:
                        data = inference_frame.data[self._model.window_size_samples :]
                        inference_frames.append(
                            AudioFrame(
                                data=data,
                                sample_rate=self._opts.sample_rate,
                                num_channels=1,
                                samples_per_channel=len(data) // 2,
                            )
                        )
        finally:
            self._is_closed = True
            await self._transcript_queue.put(None)



