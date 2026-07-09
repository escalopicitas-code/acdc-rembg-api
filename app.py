import io
import base64
import os
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from carvekit.api.high import HiInterface
from PIL import Image

app = FastAPI(title="ACDC API")

# Carrega o modelo CarveKit uma vez ao iniciar
interface = HiInterface(
    object_type="object",
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

def check_auth(request: Request):
    if not API_KEY:
        return
    key = (
        request.headers.get("x-api-key", "") or
        request.headers.get("authorization", "").replace("Bearer ", "")
    )
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def decode_base64(data: str) -> bytes:
    if "base64," in data:
        data = data.split("base64,", 1)[1]
    return base64.b64decode(data)

# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "ACDC API", "endpoints": ["/remove-background", "/extract-images"]}

@app.get("/health")
def health():
    return {"status": "ok"}

# ── Remoção de fundo (CarveKit) ──────────────────────────────────────────────

class RemoveRequest(BaseModel):
    image: str

@app.post("/remove-background")
async def remove_background(body: RemoveRequest, request: Request):
    check_auth(request)

    try:
        input_bytes = decode_base64(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagem base64 inválida.")

    try:
        img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {str(e)}")

    try:
        result = interface([img])[0].convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no CarveKit: {str(e)}")

    # Centraliza em canvas 1024x1024 com 15% padding
    W, H = result.size
    CANVAS = 1024
    MAX_SIZE = int(CANVAS * 0.70)
    scale = min(MAX_SIZE / W, MAX_SIZE / H, 1.0)
    result = result.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(result, ((CANVAS - result.width) // 2, (CANVAS - result.height) // 2), result)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    out_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        "success": True,
        "image": f"data:image/png;base64,{out_b64}",
        "provider": "carvekit",
        "message": "Fundo removido e centralizado em PNG 1024x1024."
    })

# ── Extração de imagens do PDF (PyMuPDF) ─────────────────────────────────────

class ExtractRequest(BaseModel):
    pdf: str                        # PDF inteiro em base64
    page: Optional[int] = None      # página específica (1-based); None = todas
    min_width: Optional[int] = 80   # ignorar imagens menores que isso
    min_height: Optional[int] = 80

@app.post("/extract-images")
async def extract_images(body: ExtractRequest, request: Request):
    check_auth(request)

    try:
        pdf_bytes = decode_base64(body.pdf)
    except Exception:
        raise HTTPException(status_code=400, detail="PDF base64 inválido.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF inválido: {str(e)}")

    min_w = body.min_width or 80
    min_h = body.min_height or 80

    # Páginas a processar (0-based internamente)
    if body.page is not None:
        pages = [body.page - 1]
    else:
        pages = list(range(len(doc)))

    images = []
    seen_xrefs = set()

    for page_idx in pages:
        if page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]
        img_list = page.get_images(full=True)

        for img_info in img_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png").lower()
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Filtrar imagens muito pequenas (ícones, texturas, etc.)
                if width < min_w or height < min_h:
                    continue

                # Converter para PNG normalizado
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=92)
                b64 = base64.b64encode(buf.getvalue()).decode()

                images.append({
                    "xref": xref,
                    "page": page_idx + 1,
                    "width": pil_img.width,
                    "height": pil_img.height,
                    "originalExt": ext,
                    "image": f"data:image/jpeg;base64,{b64}"
                })
            except Exception:
                continue  # ignora imagens corrompidas

    doc.close()

    return JSONResponse({
        "success": True,
        "total": len(images),
        "images": images
    })
