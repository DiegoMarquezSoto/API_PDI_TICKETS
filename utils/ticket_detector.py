# utils/ticket_detector.py
import cv2
import numpy as np
from typing import Tuple, Optional
import os


def order_points(pts):
    """
    Ordena puntos en: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left: menor suma, Bottom-right: mayor suma
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right: menor diferencia, Bottom-left: mayor diferencia
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts):
    """
    Aplica transformación de perspectiva para enderezar el ticket
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calcular ancho del ticket
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Calcular alto del ticket
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Coordenadas destino (rectángulo perfecto)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Calcular matriz de perspectiva y aplicar
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def segment_ticket_robust(image_path: str, save_debug: bool = True) -> Tuple[str, dict]:
    """
    Segmenta el ticket usando múltiples estrategias robustas
    
    ESTRATEGIAS:
    1. Detección exacta de 4 esquinas (rectángulo perfecto)
    2. MinAreaRect (rectángulo de área mínima - funciona con tickets rotados)
    3. Bounding box del contorno más grande (fallback)
    
    Args:
        image_path: Ruta de la imagen original
        save_debug: Si True, guarda imágenes de debug
    
    Returns:
        (segmented_path, info): Ruta de la imagen segmentada e información
    """
    # 1. Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    orig = image.copy()
    basename = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
    
    # 2. Redimensionar para procesamiento (mantener ratio)
    ratio = image.shape[0] / 800.0
    h = 800
    w = int(image.shape[1] / ratio)
    image_resized = cv2.resize(image, (w, h))
    
    # 3. Preprocesamiento robusto
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Blur para reducir ruido
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detección de bordes con Canny
    edged = cv2.Canny(blur, 50, 200)
    
    # Operaciones morfológicas para cerrar huecos y unir bordes
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Encontrar contornos
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # Top 10 contornos
    
    if len(cnts) == 0:
        print("No se encontraron contornos")
        return image_path, {"method": "no_segmentation", "reason": "no_contours"}
    
    target_contour = None
    method = ""
    
    # ESTRATEGIA 1: Búsqueda exacta de 4 esquinas (rectángulo perfecto)
    print("Estrategia 1: Buscando rectángulo de 4 esquinas...")
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        area = cv2.contourArea(c)
        image_area = image_resized.shape[0] * image_resized.shape[1]
        
        # Debe tener 4 lados y área significativa (al menos 10% de la imagen)
        if len(approx) == 4 and area > image_area * 0.1:
            target_contour = approx
            method = "exact_4_corners"
            print(f"✓ Encontrado: Rectángulo de 4 esquinas (área: {area:.0f})")
            break
    
    # ESTRATEGIA 2: MinAreaRect (rectángulo de área mínima - para tickets rotados)
    if target_contour is None:
        print("Estrategia 2: Usando MinAreaRect (para tickets rotados)...")
        c = cnts[0]  # Contorno más grande
        area = cv2.contourArea(c)
        image_area = image_resized.shape[0] * image_resized.shape[1]
        
        if area > image_area * 0.1:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            target_contour = np.int32(box)
            method = "min_area_rect"
            print(f"Encontrado: MinAreaRect (área: {area:.0f})")
    
    # ESTRATEGIA 3: Bounding box simple (último recurso)
    if target_contour is None:
        print("Estrategia 3: Usando bounding box del contorno más grande...")
        c = cnts[0]
        area = cv2.contourArea(c)
        image_area = image_resized.shape[0] * image_resized.shape[1]
        
        if area > image_area * 0.05:
            x, y, w_box, h_box = cv2.boundingRect(c)
            target_contour = np.array([
                [x, y],
                [x + w_box, y],
                [x + w_box, y + h_box],
                [x, y + h_box]
            ], dtype=np.int32)
            method = "bounding_box"
            print(f"Encontrado: Bounding box (área: {area:.0f})")
    
    # 5. Verificar si se encontró algo
    if target_contour is None:
        print("No se detectó ticket con ninguna estrategia")
        return image_path, {"method": "no_segmentation", "reason": "no_valid_contour"}
    
    # 6. Mapear puntos a la imagen original (alta calidad)
    if target_contour.shape[0] == 4:
        mapped_points = target_contour.reshape(4, 2) * ratio
    else:
        # Si no son 4 puntos, tomar el bounding box
        x, y, w_box, h_box = cv2.boundingRect(target_contour)
        mapped_points = np.array([
            [x, y],
            [x + w_box, y],
            [x + w_box, y + h_box],
            [x, y + h_box]
        ], dtype=np.float32) * ratio
    
    # 8. Aplicar transformación de perspectiva a la imagen original
    try:
        warped = four_point_transform(orig, mapped_points)
    except Exception as e:
        print(f"Error en transformación de perspectiva: {e}")
        # Fallback: recortar con bounding box simple
        x, y, w_box, h_box = cv2.boundingRect(target_contour)
        x, y, w_box, h_box = int(x * ratio), int(y * ratio), int(w_box * ratio), int(h_box * ratio)
        warped = orig[y:y+h_box, x:x+w_box]
        method = f"{method}_fallback"
    
    # 7. Guardar resultado final
    segmented_path = image_path.replace('.png', '_segmented.png').replace('.jpg', '_segmented.jpg')
    cv2.imwrite(segmented_path, warped)
    
    # 8. Información de la segmentación
    info = {
        "method": method,
        "original_size": f"{orig.shape[1]}x{orig.shape[0]}",
        "segmented_size": f"{warped.shape[1]}x{warped.shape[0]}",
        "area_reduction": f"{(1 - (warped.shape[0] * warped.shape[1]) / (orig.shape[0] * orig.shape[1])) * 100:.1f}%",
        "contours_analyzed": len(cnts)
    }
    
    print(f"Ticket segmentado exitosamente")
    print(f"  Método: {method}")
    print(f"  Tamaño original: {info['original_size']}")
    print(f"  Tamaño segmentado: {info['segmented_size']}")
    print(f"  Reducción: {info['area_reduction']}")
    
    return segmented_path, info


def crop_ticket_from_image(image_path: str, save_debug: bool = True) -> Tuple[str, dict]:
    """
    Función principal - Segmenta el ticket usando estrategias robustas
    
    Args:
        image_path: Ruta de la imagen original
        save_debug: Si True, guarda imágenes de debug
    
    Returns:
        (segmented_path, info): Ruta de la imagen segmentada e información
    """
    return segment_ticket_robust(image_path, save_debug)