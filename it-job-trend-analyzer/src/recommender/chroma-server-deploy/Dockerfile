# Chọn base image có Python 3.11 và chromadb cài sẵn
FROM python:3.11-slim

# Cài dependencies
RUN pip install --no-cache-dir chromadb

# Copy thư mục dữ liệu Chroma (persist_directory) lên container
COPY ./chroma /app/chroma

WORKDIR /app

# Lệnh chạy Chroma server với thư mục persist đã copy
CMD ["chroma", "run", "--persist_directory", "/app/chroma", "--host", "0.0.0.0", "--port", "8000"]
