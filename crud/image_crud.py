import cv2
import numpy as np
import base64
import datetime
import os
import io

# Almacenamiento en memoria #
processed_images = {}

# Carpeta para guardar imágenes #

IMAGES_FOLDER_FILE = "imagenes_ingresadas_como_archivo_upload"
os.makedirs(IMAGES_FOLDER_FILE, exist_ok=True)

IMAGES_FOLDER_B64 = "imagenes_decodificadas_base64"
os.makedirs(IMAGES_FOLDER_B64, exist_ok=True)

def process_uploaded_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Procesa imagen subida y retorna bytes procesados y nombre"""
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Archivo no válido")
    
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    _, buffer = cv2.imencode(".png", gray_image)
    processed_bytes = buffer.tobytes()
    
    # Generar nombre único
    name = f"processed_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
    filepath = os.path.join(IMAGES_FOLDER_FILE, name)
    with open(filepath, "wb") as f:
        f.write(processed_bytes)
    
    # Guardar en memoria
    processed_images[name] = processed_bytes
    
    return processed_bytes, name

def process_base64_image(base64_str: str) -> dict:

    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    # Decodificar base64
    image_bytes = base64.b64decode(base64_str)
    
    # Convertir a numpy array
    np_image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    
    # Generar nombre único
    filename = f"imagen_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(IMAGES_FOLDER_B64, filename)
    
    # Guardar en disco
    cv2.imwrite(filepath, img)
    
    # Guardar en memoria
    processed_images[filename] = image_bytes
    
    return {
        "status": True,
        "message": "Imagen guardada correctamente",
        "filename": filename,
        "path": filepath,
        "size": f"{len(image_bytes)} bytes"
    }

def get_all_images() -> list[dict]:
    """Retorna todas las imágenes procesadas en base64"""
    items = []
    for name, img_bytes in processed_images.items():
        b64 = base64.b64encode(img_bytes).decode("ascii")
        items.append({
            "name": name,
            "data": f"data:image/png;base64,{b64}"
        })
    return items