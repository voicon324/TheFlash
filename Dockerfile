# BASE IMAGE
# VNPT AI - The Builder Track 2
# Sử dụng CUDA 12.2 để khớp với Server BTC
# -----------------------------------------------------------
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Prevent timezone prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------
# SYSTEM DEPENDENCIES
# Cài đặt Python, Pip và các gói hệ thống cần thiết
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Link python3 thành python nếu cần
RUN ln -sf /usr/bin/python3 /usr/bin/python

# -----------------------------------------------------------
# PROJECT SETUP
# -----------------------------------------------------------
# Thiết lập thư mục làm việc
WORKDIR /code

# Copy requirements first for better caching
COPY requirements.txt /code/requirements.txt

# -----------------------------------------------------------
# INSTALL LIBRARIES
# -----------------------------------------------------------
# Nâng cấp pip và cài đặt các thư viện từ requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code vào trong container
COPY . /code

# Make inference.sh executable
RUN chmod +x /code/inference.sh

# -----------------------------------------------------------
# EXECUTION
# Lệnh chạy mặc định khi container khởi động
# Pipeline sẽ đọc private_test.json và xuất ra submission.csv, submission_time.csv
# -----------------------------------------------------------
CMD ["bash", "inference.sh"]
