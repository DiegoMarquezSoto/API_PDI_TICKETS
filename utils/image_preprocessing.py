# utils/image_preprocessing.py
import cv2
import numpy as np
from typing import Optional


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Mejora el contraste local sin amplificar ruido
    
    Args:
        image: Imagen en escala de grises
        clip_limit: Límite de contraste (2.0-4.0 recomendado)
        tile_grid_size: Tamaño de grid para tiles (8x8 o 16x16)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Reduce ruido manteniendo bordes nítidos
    
    Args:
        image: Imagen en escala de grises
        strength: Fuerza del denoising (7-15 recomendado)
    """
    return cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def sharpen_image(image: np.ndarray, kernel_size: int = 3, amount: float = 1.5) -> np.ndarray:
    """
    Aumenta la nitidez de bordes y texto
    
    Args:
        image: Imagen en escala de grises
        kernel_size: Tamaño del kernel (3 o 5)
        amount: Intensidad del sharpening (1.0-2.5)
    """
    # Crear kernel de sharpening
    if kernel_size == 3:
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
    else:
        kernel = np.array([[-1, -1, -1, -1, -1],
                          [-1,  2,  2,  2, -1],
                          [-1,  2,  8,  2, -1],
                          [-1,  2,  2,  2, -1],
                          [-1, -1, -1, -1, -1]]) / 8.0
    
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Blend con original según amount
    if amount != 1.0:
        sharpened = cv2.addWeighted(image, 1 - (amount - 1), sharpened, amount, 0)
    
    return sharpened


def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Binarización adaptativa - útil para iluminación irregular
    
    Args:
        image: Imagen en escala de grises
        block_size: Tamaño de vecindad (11, 15, 21)
        c: Constante de ajuste (2-10)
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )


def preprocess_for_ocr(
    image_path: str,
    output_path: Optional[str] = None,
    enable_clahe: bool = True,
    enable_denoise: bool = True,
    enable_sharpen: bool = True,
    enable_threshold: bool = False,  # Solo si imagen muy borrosa
    clahe_clip: float = 2.5,
    denoise_strength: int = 10,
    sharpen_amount: float = 1.5
) -> str:
    """
    Pipeline completo de preprocesamiento para mejorar OCR
    
    Args:
        image_path: Ruta de imagen segmentada
        output_path: Ruta de salida (opcional, default: _preprocessed.png)
        enable_*: Activar/desactivar cada técnica
        
    Returns:
        Ruta de la imagen preprocesada
    """
    print("Iniciando preprocesamiento avanzado...")
    
    # 1. Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo leer: {image_path}")
    
    # 2. Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = gray.copy()
    
    # 3. CLAHE - Mejora contraste local
    if enable_clahe:
        print(f"  Aplicando CLAHE (clip={clahe_clip})...")
        processed = apply_clahe(processed, clip_limit=clahe_clip)
    
    # 4. Denoising - Reduce ruido
    if enable_denoise:
        print(f"  Aplicando denoising (strength={denoise_strength})...")
        processed = denoise_image(processed, strength=denoise_strength)
    
    # 5. Sharpening - Mejora nitidez
    if enable_sharpen:
        print(f"  Aplicando sharpening (amount={sharpen_amount})...")
        processed = sharpen_image(processed, amount=sharpen_amount)
    
    # 6. Binarización adaptativa (opcional - solo para casos extremos)
    if enable_threshold:
        print("  Aplicando binarización adaptativa...")
        processed = adaptive_threshold(processed)
    
    # 7. Guardar resultado
    if output_path is None:
        output_path = image_path.replace('.png', '_preprocessed.png').replace('.jpg', '_preprocessed.jpg')
    
    cv2.imwrite(output_path, processed)
    print(f"Preprocesamiento completado: {output_path}")
    
    return output_path