import os
import tarfile
import zipfile
import urllib.request
import shutil
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_URLS = {
    "vi": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2",

    "en": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",

    "zh": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",

    "ja": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2",

    "ko": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2",

    "fr": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2",

    "de": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2",

    "es": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2",

    "ru": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2",

    "ar": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2",
}

def download_and_extract(lang, url):
    target_folder = MODELS_DIR / f"zipformer-{lang}"
    if target_folder.exists() and any(target_folder.iterdir()):
        print(f"[SKIP] Mô hình [{lang.upper()}] đã tồn tại trong models/zipformer-{lang}")
        return
    archive_name = url.split("/")[-1]
    archive_path = MODELS_DIR / archive_name

    print(f"[DOWNLOAD] Đang tải mô hình [{lang.upper()}]...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[EXTRACT] Đang giải nén {archive_name}...")
        extracted_dir_name = None
        if archive_name.endswith(".tar.bz2") or archive_name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=MODELS_DIR)
                extracted_dir_name = tar.getnames()[0].split('/')[0]
        elif archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(MODELS_DIR)
                extracted_dir_name = zip_ref.namelist()[0].split('/')[0]
        if extracted_dir_name:
            extracted_path = MODELS_DIR / extracted_dir_name
            if extracted_path.exists() and extracted_path != target_folder:
                if target_folder.exists():
                    shutil.rmtree(target_folder)
                extracted_path.rename(target_folder)
        if archive_path.exists():
            archive_path.unlink()
        print(f"[OK] Đã hoàn tất cài đặt [{lang.upper()}] vào models/zipformer-{lang}!\n")
    except Exception as e:
        print(f"[LỖI] Không thể tải/giải nén mô hình [{lang.upper()}]: {e}\n")

if __name__ == "__main__":
    for lang, url in MODEL_URLS.items():
        download_and_extract(lang, url)
    print("HOÀN TẤT TẤT CẢ!")