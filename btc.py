import os
import json
import base64
from datetime import datetime
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
import base64, io, hashlib


API_KEY = "sk-ug1poexeERrICjPNvLSooQ" #os.getenv("THUCCHIEN_API_KEY", "sk-ug1poexeERrICjPNvLSooQ")
URL = "https://api.thucchien.ai/gemini/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
# API_KEY = os.getenv("THUCCHIEN_API_KEY", "AIzaSyA3yyChOko7uZJdU3M0v9d6zUqxvdaDC6Q")
# URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"

def image_to_base64(image: Image.Image, format="PNG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def image_to_base64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_image(path_str):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    # giữ đúng xoay EXIF, chuyển RGBA cho hiển thị sắc nét
    img = ImageOps.exif_transpose(Image.open(p)).convert("RGBA")
    raw = p.read_bytes()
    sha12 = hashlib.sha256(raw).hexdigest()[:12]
    print(f"Loaded: {p.name} | size: {img.size} | bytes: {len(raw)/1024:.1f} KB | sha256[:12]={sha12}")
    return img


my_prompt_25 = """
Ghép chữ vào hình ảnh tạo 1 banner quảng cáo.
"""

p25 = r"1.png"
image_input_p25 = load_image(p25)
image_input_p25_b64 = image_to_base64(image_input_p25)

p26 = r"Screenshot 2025-10-25 at 13.57.31.png"
image_input_p26 = load_image(p26)
image_input_p26_b64 = image_to_base64(image_input_p26)

payload = {
    "contents": [
        {
            "parts": [
               
                {"text": my_prompt_25},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": image_input_p25_b64,
                    },
                },
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": image_input_p26_b64,
                    },
                },
            ]
        }
    ],
    "generationConfig": {
        'responseModalities': ["Image"],
        'candidateCount': 1,
        "imageConfig": {"aspectRatio": "16:9"}
    },
}

headers = {
    "x-goog-api-key": API_KEY,  # native Gemini dùng header này
    "Content-Type": "application/json",
}

# Gọi API
resp = requests.post(URL, headers=headers, data=json.dumps(payload))
print("HTTP", resp.status_code)

# Helper: lưu + hiển thị ảnh từ inlineData
def save_and_show_inline_image(
    b64_data: str, mime_type: str = "image/png", prefix: str = "gemini_image"
):
    # Chọn phần mở rộng dựa vào mime
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
    }
    ext = ext_map.get(mime_type.lower(), "png")

    img_bytes = base64.b64decode(b64_data)

    # Tên file theo thời gian thực
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.{ext}"
    Path(fname).write_bytes(img_bytes)

    print(f"Saved: {fname}")

if resp.ok:
    data = resp.json()
    # Trích tất cả inlineData trong candidates -> content.parts
    found = 0
    for cand in data.get("candidates", []):
        content = cand.get("content", {})
        for part in content.get("parts", []):
            inline = part.get("inlineData") or part.get(
                "inline_data"
            )  # phòng khi viết khác case
            if inline and inline.get("data"):
                save_and_show_inline_image(
                    b64_data=inline["data"],
                    mime_type=inline.get("mimeType", "image/png"),
                    prefix="gemini_image",
                )
                found += 1
    if found == 0:
        print("No inline image found in response.")
        print(json.dumps(data, indent=2))
else:
    # In lỗi chi tiết để chẩn đoán
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)
