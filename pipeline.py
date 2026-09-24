import os
import json
import base64
import asyncio
import urllib.request
import urllib.parse
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import edge_tts
import sherpa_onnx

SAMPLE_RATE = 16000
WINDOW = 512

EDGE_TTS_VOICES = {
    "vi": "vi-VN-NamMinhNeural",
    "en": "en-US-AndrewNeural",
    "zh": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "th": "th-TH-NiwatNeural",
    "id": "id-ID-ArdiNeural",
    "hi": "hi-IN-MadhurNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-KillianNeural",
    "es": "es-ES-AlvaroNeural",
    "it": "it-IT-DiegoNeural",
    "ru": "ru-RU-DmitryNeural",
    "pt": "pt-BR-AntonioNeural",
    "ar": "ar-SA-HamedNeural"
}

executor = ThreadPoolExecutor(max_workers=2)

class Engine:

    def __init__(self, root="."):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.sherpa = sherpa_onnx
        
        # Đặt 2 threads để tránh làm Render bị đơ CPU
        self.threads = int(os.getenv("NUM_THREADS", "2"))
        torch.set_num_threads(self.threads)
        self.recognizers = {}
        print("[INIT] Engine Zipformer đã khởi tạo thành công (Load lazy khi có kết nối)!")

    def get_recognizer(self, lang="vi"):
        # 1. Đảm bảo file silero_vad.onnx luôn sẵn sàng
        vad_path = self.models_dir / "silero_vad.onnx"
        if not vad_path.exists():
            print("[INFO] Đang tải silero_vad.onnx vào models/...")
            self.models_dir.mkdir(exist_ok=True, parents=True)
            url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
            urllib.request.urlretrieve(url, vad_path)

        if lang in self.recognizers:
            return self.recognizers[lang]
        
        folder = self.models_dir / f"zipformer-{lang}"
        if not folder.exists() and lang != "vi":
            folder = self.models_dir / "zipformer-vi"
            
        encoder_files = list(folder.glob("encoder*.onnx"))
        decoder_files = list(folder.glob("decoder*.onnx"))
        joiner_files = list(folder.glob("joiner*.onnx"))
        tokens_file = folder / "tokens.txt"
        
        if encoder_files and decoder_files and joiner_files and tokens_file.exists():
            print(f"[LOAD] Nạp Zipformer [{lang}] từ models/{folder.name}...")
            recognizer = self.sherpa.OfflineRecognizer.from_transducer(
                tokens=str(tokens_file),
                encoder=str(encoder_files[0]),
                decoder=str(decoder_files[0]),
                joiner=str(joiner_files[0]),
                num_threads=self.threads, 
                sample_rate=SAMPLE_RATE, 
                feature_dim=80,
                decoding_method="greedy_search", 
                provider="cpu"
            )
            self.recognizers[lang] = recognizer
            return recognizer
        else:
            raise FileNotFoundError(
                f"[ERROR] Không tìm thấy đủ bộ file ONNX trong thư mục: {folder.resolve()}"
            )

    def new_vad(self):
        vad_path = self.models_dir / "silero_vad.onnx"
        if not vad_path.exists():
            self.get_recognizer("vi")  # Trigger tự tải VAD nếu chưa có

        config = self.sherpa.VadModelConfig()
        config.silero_vad.model = str(vad_path)
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.4 
        config.silero_vad.min_speech_duration = 0.25   
        config.sample_rate = SAMPLE_RATE
        return self.sherpa.VoiceActivityDetector(config, buffer_size_in_seconds=300)

    def decode(self, samples, lang="vi"):
        if len(samples) < 1600:
            return ""
        recognizer = self.get_recognizer(lang)
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.asarray(samples, dtype=np.float32))
        recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def translate(self, text, src_lang="vi", target_lang="en"):
        if not text.strip() or src_lang == target_lang:
            return text
        try:
            query = urllib.parse.quote(text)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_lang}&tl={target_lang}&dt=t&q={query}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=3.0) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                if res_data and res_data[0]:
                    translated_text = "".join([item[0] for item in res_data[0] if item[0]])
                    return translated_text.strip()
            return text
        except Exception:
            return text

    async def generate_tts(self, text, target_lang):
        if not text.strip():
            return ""
        voice = EDGE_TTS_VOICES.get(target_lang, "en-US-AndrewNeural")
        try:
            communicate = edge_tts.Communicate(text, voice, rate="+25%")  
            audio_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.extend(chunk["data"])
            return base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            print("[TTS ERROR]", e)
            return ""


class Session:

    def __init__(self, engine, src_lang="vi", target_lang="en"):
        self.engine = engine
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.vad = engine.new_vad()
        self.pending = np.empty(0, dtype=np.float32)
        self.speech_queue = Queue()
        self.unsegmented_buffer = np.empty(0, dtype=np.float32)
        self.full_src_history = []
        self.full_tgt_history = []
        self.last_partial_text = ""

    def update_config(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang

    async def feed(self, samples):
        if len(samples) == 0 or np.max(np.abs(samples)) < 0.01:
            return []
        self.pending = np.concatenate((self.pending, samples))
        self.unsegmented_buffer = np.concatenate((self.unsegmented_buffer, samples))
        events = []
        loop = asyncio.get_event_loop()
        while self.pending.size >= WINDOW:
            chunk = self.pending[:WINDOW].copy()
            self.pending = self.pending[WINDOW:]
            self.vad.accept_waveform(chunk)
            while not self.vad.empty():
                speech_samples = np.array(self.vad.front.samples, dtype=np.float32, copy=True)
                self.vad.pop()
                self.speech_queue.put(speech_samples)
        while not self.speech_queue.empty():
            speech_chunk = self.speech_queue.get()
            final_text = await loop.run_in_executor(
                executor, self.engine.decode, speech_chunk, self.src_lang
            )
            if final_text:
                trans = await loop.run_in_executor(
                    executor, self.engine.translate, final_text, self.src_lang, self.target_lang
                )
                self.full_src_history.append(final_text)
                self.full_tgt_history.append(trans)
                self.last_partial_text = "" 
                events.append({
                    "type": "final", 
                    "text": final_text, 
                    "translation": trans,
                    "full_src": " ".join(self.full_src_history),
                    "full_tgt": " ".join(self.full_tgt_history),
                    "source_lang": self.src_lang, 
                    "target_lang": self.target_lang
                })
            self.unsegmented_buffer = np.empty(0, dtype=np.float32)
        if len(events) == 0 and self.unsegmented_buffer.size >= 8000:
            partial_text = await loop.run_in_executor(
                executor, self.engine.decode, self.unsegmented_buffer, self.src_lang
            )
            if partial_text and partial_text != self.last_partial_text:
                self.last_partial_text = partial_text
                events.append({
                    "type": "partial",
                    "text": partial_text,
                    "full_src": (" ".join(self.full_src_history) + " " + partial_text).strip(),
                    "source_lang": self.src_lang
                })
            
            self.unsegmented_buffer = self.unsegmented_buffer[-1600:]
        return events

    async def force_flush(self):
        events = []
        loop = asyncio.get_event_loop()
        silence_padding = np.zeros(8000, dtype=np.float32)
        self.pending = np.concatenate((self.pending, silence_padding))
        self.unsegmented_buffer = np.concatenate((self.unsegmented_buffer, silence_padding))
        while self.pending.size >= WINDOW:
            chunk = self.pending[:WINDOW].copy()
            self.pending = self.pending[WINDOW:]
            self.vad.accept_waveform(chunk)
            while not self.vad.empty():
                speech_samples = np.array(self.vad.front.samples, dtype=np.float32, copy=True)
                self.vad.pop()
                self.speech_queue.put(speech_samples)
        while not self.speech_queue.empty():
            speech_chunk = self.speech_queue.get()
            final_text = await loop.run_in_executor(
                executor, self.engine.decode, speech_chunk, self.src_lang
            )
            if final_text:
                trans = await loop.run_in_executor(
                    executor, self.engine.translate, final_text, self.src_lang, self.target_lang
                )
                self.full_src_history.append(final_text)
                self.full_tgt_history.append(trans)
                events.append({
                    "type": "final", 
                    "text": final_text, 
                    "translation": trans,
                    "full_src": " ".join(self.full_src_history),
                    "full_tgt": " ".join(self.full_tgt_history),
                    "source_lang": self.src_lang, 
                    "target_lang": self.target_lang
                })
        if len(events) == 0 and self.unsegmented_buffer.size >= 1600:
            final_text = await loop.run_in_executor(
                executor, self.engine.decode, self.unsegmented_buffer, self.src_lang
            )
            if final_text:
                trans = await loop.run_in_executor(
                    executor, self.engine.translate, final_text, self.src_lang, self.target_lang
                )
                self.full_src_history.append(final_text)
                self.full_tgt_history.append(trans)
                events.append({
                    "type": "final", 
                    "text": final_text, 
                    "translation": trans,
                    "full_src": " ".join(self.full_src_history),
                    "full_tgt": " ".join(self.full_tgt_history),
                    "source_lang": self.src_lang, 
                    "target_lang": self.target_lang
                })
        self.pending = np.empty(0, dtype=np.float32)
        self.unsegmented_buffer = np.empty(0, dtype=np.float32)
        self.last_partial_text = ""
        self.vad.reset()
        return events