FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu carvekit==4.1.2 fastapi==0.115.0 uvicorn==0.30.6 python-multipart==0.0.9 Pillow==10.4.0 PyMuPDF==1.24.5
COPY app.py .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
