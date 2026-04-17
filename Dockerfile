FROM informaticsmatters/rdkit-python3-debian:Release_2024_09

WORKDIR /app

# 使用 python -m pip 安装，完整路径
RUN python -m pip install --no-cache-dir --break-system-packages \
    Flask \
    gunicorn \
    werkzeug \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    matplotlib \
    seaborn

# 复制应用代码
COPY . .

# 创建上传目录
RUN mkdir -p static/uploads

# 暴露端口
EXPOSE 8080

# 使用完整路径运行 gunicorn
CMD ["python", "-m", "gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2"]
