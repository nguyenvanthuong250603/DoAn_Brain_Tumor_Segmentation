FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng cache của Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- PHẦN SỬA ĐỔI QUAN TRỌNG ---

# Copy toàn bộ mã nguồn hiện tại (trong thư mục 'code') vào '/app' trong container
# Bao gồm: app.py, models/, Data_Test/, process_image.py...
COPY . .

# Nếu muốn copy riêng lẻ (an toàn hơn) thì dùng:
# COPY app.py .
# COPY process_image.py .
# COPY model.py .
# COPY models ./models
# COPY Data_Test ./Data_Test

# Mở port
EXPOSE 8000

# Chạy file app.py (vì nó nằm ngay thư mục gốc /app)
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]