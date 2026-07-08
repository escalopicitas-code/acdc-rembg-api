FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU primeiro (pacote grande, melhor cachear separado)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Demais dependências
COPY requirements.txt .
RUN pip install --no-cache-dir \
    carvekit==4.1.0 \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    python-multipart==0.0.9 \
    Pillow==10.4.0

# Pré-baixa o modelo tracer_b7 para não baixar no primeiro request
RUN python -c "
from carvekit.api.high import HiInterface
HiInterface(object_type='object', batch_size_seg=1, batch_size_matting=1, device='cpu', seg_mask_size=640, matting_mask_size=2048, fp16=False)
print('Modelo tracer_b7 carregado com sucesso.')
"

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
