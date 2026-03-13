import easyocr
from typing import Dict
import numpy as np


reader = easyocr.Reader(['es', 'en'], gpu=True) #Config inicial de OCR (idiomas y uso de GPU)

def convert_to_native_types(obj):
    ### conversion de numpu a typs nativos ###
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

### EXTRACCION DE TEXTOS ###
def extract_text_from_image(image_path: str, min_confidence: float = 0.2) -> Dict:
    
    try:
        results = reader.readtext(image_path, 
            # Mejora de contraste #
            contrast_ths=0.2,        
            adjust_contrast=0.5,     
            
            # Filtrado para mejorar el procesamiento #
            text_threshold=0.5,      
            low_text=0.3,           
            link_threshold=0.4,      
            canvas_size=1920,        
            mag_ratio=1.75            
        )
        
        lines = []
        for (bbox, text, prob) in results:
            if float(prob) >= min_confidence: 
                lines.append({
                    "text": str(text),
                    "confidence": float(prob),
                    "bbox": convert_to_native_types(bbox)
                })
        
        # Solo texto con confianza suficiente
        text_full = " ".join([line["text"] for line in lines])
        
        return {
            "success": True,
            "text_full": text_full,
            "lines": lines,
            "total_lines": len(lines),
            "filtered_count": len(results) - len(lines)  # Cuántos se filtraron
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text_full": "",
            "lines": [],
            "total_lines": 0,
            "filtered_count": 0
        }