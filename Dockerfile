FROM python:3.10

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY ./app /code/app
COPY ./src /code/src
COPY ./model /code/model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]