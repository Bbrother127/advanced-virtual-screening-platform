# 使用完整的 Python 镜像以避免 RDKit 依赖问题
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 安装系统依赖（RDKit 需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    librdkit-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建上传目录
RUN mkdir -p static/uploads

# 暴露端口
EXPOSE 5000

# 使用 gunicorn 运行应用
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]