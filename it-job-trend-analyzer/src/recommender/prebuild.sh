#!/bin/bash

echo "Starting model pre-download process for SentenceTransformer..."

# Định nghĩa thư mục để lưu trữ models trong thư mục gốc của project
# RENDER_PROJECT_ROOT là biến môi trường Render cung cấp
MODEL_CACHE_DIR="/opt/render/project/src/.cache/huggingface/hub" # Thư mục cache mặc định của Hugging Face
mkdir -p "$MODEL_CACHE_DIR" # Đảm bảo thư mục này tồn tại

# Xuất biến môi trường HF_HOME để SentenceTransformer lưu vào đó
export HF_HOME="$MODEL_CACHE_DIR"
export SENTENCE_TRANSFORMERS_HOME="$MODEL_CACHE_DIR" # Đảm bảo cho SBERT cũng dùng cache này

# Sử dụng lệnh python hoặc python3 tùy theo hệ thống
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found! Please ensure Python is installed."
    exit 1
fi

# Dòng này sẽ tải model 'sentence-transformers/paraphrase-mpnet-base-v2' vào thư mục cache đã chỉ định
# (Lưu ý: không cần chỉ định `cache_folder` ở đây vì đã dùng HF_HOME)
$PYTHON_CMD -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2'); print('Model paraphrase-mpnet-base-v2 pre-downloaded or found in cache.')"

echo "Model pre-download process finished."