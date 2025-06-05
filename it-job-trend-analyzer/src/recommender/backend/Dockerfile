# Sử dụng image Python 3.10 dựa trên Debian Bookworm
FROM python:3.10-bookworm

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Tạo thư mục cache bên trong /app và cấp quyền đầy đủ
RUN mkdir -p /app/model_cache && chmod -R 777 /app/model_cache
# Không cần chown 1000:1000 nếu chúng ta chạy với USER root
# RUN chown -R 1000:1000 /app/model_cache

# Copy các file cần thiết để cài đặt
COPY requirements_final.txt .
COPY backend_deploy.py .

# Cài đặt các thư viện Python từ requirements_final.txt
RUN pip install --no-cache-dir -r requirements_final.txt

# Thiết lập biến môi trường cho Hugging Face Hub và SentenceTransformers
# Dù model không được tải trong build, các biến này vẫn hướng dẫn thư viện nơi lưu cache khi runtime.
ENV HF_HOME /app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME /app/model_cache
ENV TRANSFORMERS_CACHE /app/model_cache

# **XÓA BỎ DÒNG RUN NÀY (sẽ tải model trong runtime):**
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2', cache_folder='/app/model_cache'); print('Model paraphrase-mpnet-base-v2 pre-downloaded or found in cache.')"

# Expose cổng mà Uvicorn sẽ lắng nghe. Hugging Face Spaces thường yêu cầu cổng 7860.
EXPOSE 7860

# Chạy CMD với user root để đảm bảo quyền truy cập tối đa khi ứng dụng chạy
USER root

# Lệnh để khởi động ứng dụng FastAPI của bạn bằng Uvicorn
CMD ["uvicorn", "backend_deploy:app", "--host", "0.0.0.0", "--port", "7860"]