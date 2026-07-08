import io
import base64
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from carvekit.api.high import HiInterface
from PIL import Image

app = FastAPI(title="ACDC Rembg API")

# Carrega o modelo uma vez ao iniciar
# tracer_b7 = melhor para objetos/produtos/móveis
# seg_mask_size=640 = boa qualidade sem explodir a RAM
interface = HiInterface(
    object_type="object",          # otimizado para objetos, não pessoas
    batch_size_seg=1,
    batch_size_matting=1,
    device="cpu",
    seg_mask_size=640,
    matting_mask_size=2048,
    trimap_dilation=15,
    trimap_erosion_iters=5,
    fp16=False
)

API_KEY = os.environ.get("REMBG_API_KEY", "")

class RemoveRequest(BaseModel):
    image: str  # base64 com ou sem prefixo data:...

def check_auth(request: Request):
    if not API_KEY:
        return
    key = (
        request.headers.get("x-api-key", "") or
        request.headers.get("authorization", "").replace("Bearer ", "")
    )
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/")
def root():
    return {"status": "ok", "service": "ACDC CarveKit API", "model": "tracer_b7"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/remove-background")
async def remove_background(body: RemoveRequest, request: Request):
    check_auth(request)

    # Decodifica base64
    raw = body.image
    if "base64," in raw:
        raw = raw.split("base64,", 1)[1]
    try:
        input_bytes = base64.b64decode(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagem base64 inválida.")

    # Abre a imagem
    try:
        img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {str(e)}")

    # Remove fundo via CarveKit
    try:
        images_without_bg = interface([img])
        result = images_without_bg[0].convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no CarveKit: {str(e)}")

    # Centraliza em canvas 1024x1024 com 15% padding
    try:
        W, H = result.size
        CANVAS = 1024
        MAX_SIZE = int(CANVAS * 0.70)  # 70% = 15% padding cada lado
        scale = min(MAX_SIZE / W, MAX_SIZE / H, 1.0)
        new_w = int(W * scale)
        new_h = int(H * scale)
        result = result.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        left = (CANVAS - new_w) // 2
        top  = (CANVAS - new_h) // 2
        canvas.paste(result, (left, top), result)
        buf = io.BytesIO()
        canvas.save(buf, format="PNG", optimize=True)
        output_bytes = buf.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na centralização: {str(e)}")

    out_b64 = base64.b64encode(output_bytes).decode()
    return JSONResponse({
        "success": True,
        "image": f"data:image/png;base64,{out_b64}",
        "provider": "carvekit",
        "message": "Fundo removido pelo CarveKit (tracer_b7) e centralizado em PNG 1024x1024."
    })
