# 🎤 Streaming ASR for Vietnamese using Whisper

This project implements a **streaming speech-to-text (ASR)** system using a **fine-tuned Whisper-small model** for Vietnamese. The model is capable of producing transcriptions with **punctuation and proper casing**

The ASR model has been converted to **CTranslate2** format to reduce inference latency.

👉 Try the model here: [https://huggingface.co/duonguyen/whisper-vietnamese-ct2](https://huggingface.co/duonguyen/whisper-vietnamese-ct2)

---

## 🧠 Streaming Technique Overview

Since Whisper was not originally designed for streaming ASR, this project adopts a **streaming adapter technique** inspired by the LiveKit framework. The main idea is to use **voice activity detection (VAD)** — specifically **Silero VAD** — to detect and manage speech chunks in real time.

### ⏱️ Streaming mechanism:

1. Audio is continuously buffered from the microphone or audio file.
2. **Silero VAD** is applied to detect whether each incoming buffer contains speech.
3. Buffers labeled as speech are accumulated until one of the following conditions is met:
   - **Condition 1**:
     - The accumulated speech buffer exceeds a certain threshold (e.g., 0.5s), **AND**
     - The current buffer is labeled as **non-speech**, **AND**
     - The silence duration also exceeds a threshold (e.g., 0.5s).
   - **Condition 2**:
     - The accumulated buffer reaches the maximum duration that the ASR model can process (e.g., 30s for Whisper).

4. Once either condition is satisfied, the speech buffer is sent to the ASR model for inference.

### 💬 Real-time partial transcription:

To improve the real-time experience, **partial transcription** is triggered every ~1s when the buffer grows too long. These intermediate results are **not considered final** but are returned early for display purposes only.

---

## 🚀 Getting Started

### Clone the repository:

```bash
git clone https://github.com/dinhduongnguyen/Streaming-ASR-Vietnamese.git
cd Streaming-ASR-Vietnamese
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the ASR streaming:

- To run streaming from an audio file:

```bash
python infer_asr_model.py
```

- To run streaming from an audio file:

```bash
python file_streaming.py --audio_path "replace with your audio file path"
```

- To run streaming from your microphone:

```bash
python mic_streaming.py
```

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [LiveKit Agent](https://github.com/livekit/agents)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
