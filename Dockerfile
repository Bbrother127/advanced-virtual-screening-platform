FROM informaticsmatters/rdkit-python3-debian:Release_2024_09

WORKDIR /app

RUN pip install --no-cache-dir --break-system-packages \
    Flask \
    gunicorn \
    werkzeug \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    matplotlib \
    seaborn

COPY . .

RUN mkdir -p static/uploads

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
