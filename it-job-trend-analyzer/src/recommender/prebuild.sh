#!/bin/bash

echo "Starting model pre-download process for SentenceTransformer..."

# Sử dụng lệnh python hoặc python3 tùy theo hệ thống
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found! Please ensure Python is installed."
    exit 1
fi

# Dòng này sẽ tải model 'sentence-transformers/paraphrase-mpnet-base-v2' vào cache
# Nếu model đã có, nó sẽ không tải lại mà chỉ đảm bảo model sẵn sàng trong cache.
$PYTHON_CMD -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2'); print('Model paraphrase-mpnet-base-v2 pre-downloaded or found in cache.')"

echo "Model pre-download process finished."