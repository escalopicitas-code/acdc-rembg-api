import io
import base64
import os
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from carvekit.api.high import HiInterface
from PIL import Image

app = FastAPI(title="ACDC API")

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

@app.get("/")
def root():
    return {"status": "ok", "service": "ACDC API", "endpoints": ["/remove-background", "/extract-images"]}

@app.get("/health")
def health():
    return {"status": "ok"}

# ── Remoção de fundo ─────────────────────────────────────────────────────────

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

# ── Extração de imagens ───────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    # Aceita imagem da página renderizada (JPEG/PNG do canvas) OU PDF completo
    image: Optional[str] = None   # print da página renderizada — preferido
    pdf: Optional[str] = None     # PDF completo em base64 — fallback
    page: Optional[int] = None    # página (1-based), só para PDF
    min_width: Optional[int] = 80
    min_height: Optional[int] = 80

def extract_from_pil(page_img: Image.Image, min_w: int, min_h: int):
    """Extrai regiões significativas de uma imagem usando PyMuPDF via conversão."""
    # Converte PIL → bytes → fitz document de página única
    buf = io.BytesIO()
    page_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Usa fitz para abrir como imagem e extrair sub-imagens embutidas
    doc = fitz.open(stream=img_bytes, filetype="png")
    images = []
    for page in doc:
        img_list = page.get_images(full=True)
        for img_info in img_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                iw = base_image.get("width", 0)
                ih = base_image.get("height", 0)
                if iw < min_w or ih < min_h:
                    continue
                pil = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                b = io.BytesIO()
                pil.save(b, format="JPEG", quality=92)
                images.append({
                    "xref": xref, "page": 1,
                    "width": pil.width, "height": pil.height,
                    "originalExt": base_image.get("ext", "png"),
                    "image": f"data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}"
                })
            except Exception:
                continue
    doc.close()
    return images

def extract_from_pdf(pdf_bytes: bytes, page_num: Optional[int], min_w: int, min_h: int):
    """Extrai imagens embutidas diretamente do PDF via PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page_num - 1] if page_num else list(range(len(doc)))
    images = []
    seen = set()
    for pi in pages:
        if pi < 0 or pi >= len(doc):
            continue
        for img_info in doc[pi].get_images(full=True):
            xref = img_info[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                base_image = doc.extract_image(xref)
                iw = base_image.get("width", 0)
                ih = base_image.get("height", 0)
                if iw < min_w or ih < min_h:
                    continue
                pil = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                b = io.BytesIO()
                pil.save(b, format="JPEG", quality=92)
                images.append({
                    "xref": xref, "page": pi + 1,
                    "width": pil.width, "height": pil.height,
                    "originalExt": base_image.get("ext", "png"),
                    "image": f"data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}"
                })
            except Exception:
                continue
    doc.close()
    return images

@app.post("/extract-images")
async def extract_images(body: ExtractRequest, request: Request):
    check_auth(request)
    min_w = body.min_width or 80
    min_h = body.min_height or 80

    # MODO 1: imagem do canvas (print da página) — mais rápido, menor payload
    if body.image:
        try:
            img_bytes = decode_base64(body.image)
            page_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Imagem inválida: {str(e)}")
        images = extract_from_pil(page_img, min_w, min_h)
        # Se PyMuPDF não achou imagens embutidas na imagem renderizada,
        # tenta segmentar a página toda como uma única imagem
        if not images:
            b = io.BytesIO()
            page_img.save(b, format="JPEG", quality=92)
            images = [{
                "xref": 0, "page": 1,
                "width": page_img.width, "height": page_img.height,
                "originalExt": "jpg",
                "image": f"data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}"
            }]
        return JSONResponse({"success": True, "total": len(images), "images": images, "source": "canvas"})

    # MODO 2: PDF completo — fallback
    if body.pdf:
        try:
            pdf_bytes = decode_base64(body.pdf)
        except Exception:
            raise HTTPException(status_code=400, detail="PDF base64 inválido.")
        try:
            images = extract_from_pdf(pdf_bytes, body.page, min_w, min_h)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao processar PDF: {str(e)}")
        return JSONResponse({"success": True, "total": len(images), "images": images, "source": "pdf"})

    raise HTTPException(status_code=400, detail="Envie 'image' (canvas) ou 'pdf' (base64).")
