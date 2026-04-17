FROM mcs07/rdkit:2024.03.1

WORKDIR /app

RUN pip install --no-cache-dir Flask==2.3.2 gunicorn==21.2.0 werkzeug==2.3.2

COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 joblib==1.3.1 matplotlib==3.7.2 seaborn==0.12.2 || true

COPY . .

RUN mkdir -p static/uploads

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
