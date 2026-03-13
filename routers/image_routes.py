# routers/image_routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import os
from schemas.image_schema import Base64ImageRequest, ImagesListResponse
from crud.image_crud import process_uploaded_image, process_base64_image, get_all_images
from utils.ocr_utils import extract_text_from_image
from utils.llm_ticket_extractor import extract_with_ollama  # ← NUEVO
from utils.ticket_detector import crop_ticket_from_image
from utils.super_resolution import upscale_image_x4
from utils.image_preprocessing import preprocess_for_ocr

router = APIRouter(prefix="/imagenes", tags=["Imágenes"])

@router.post("/adquirir-imagen/")
async def adquirir_imagen(
    file: UploadFile = File(..., description="Sube una imagen (PNG/JPEG)", media_type="image/*"),
    auto_crop: bool = Query(True, description="Detectar y recortar ticket automáticamente"),
    enable_upscaling: bool = Query(False, description="Super-resolution x4 (lento, solo para imágenes pequeñas)"),
    enable_preprocessing: bool = Query(True, description="Aplicar preprocesamiento CLAHE/Denoise/Sharpen")
):
    """
    Pipeline completo de procesamiento de tickets:
    1. Segmentar ticket (auto_crop)
    2. Super-resolution x4 (enable_upscaling) - OPCIONAL
    3. Preprocesamiento CLAHE/Denoise/Sharpen (enable_preprocessing)
    4. OCR con EasyOCR
    5. Extracción de campos con LLM (Ollama)
    
    Parámetros:
    - auto_crop: Detecta bordes del ticket y recorta (recomendado: True)
    - enable_upscaling: Aumenta resolución x4 con EDSR (solo para imágenes <500px)
    - enable_preprocessing: Mejora contraste y nitidez (recomendado: True)
    """
    try:
        image_bytes = await file.read()
        await file.close()
        
        # Guardar la imagen original
        _, name = process_uploaded_image(image_bytes)
        image_path = os.path.join("imagenes_ingresadas_como_archivo_upload", name)
        
        # PASO 1: Detectar y recortar ticket
        crop_info = {"enabled": False}
        if auto_crop:
            try:
                cropped_path, crop_info = crop_ticket_from_image(image_path, save_debug=False)
                print(f"Ticket detectado: {crop_info.get('method')}")
                image_path = cropped_path
                crop_info["enabled"] = True
            except Exception as e:
                print(f"Error al recortar: {e}")
                crop_info = {"enabled": False, "error": str(e)}
        
        # PASO 2: Super-resolution x4 (OPCIONAL - lento)
        upscale_info = {"enabled": False}
        if enable_upscaling:
            try:
                upscaled_path = upscale_image_x4(image_path)
                image_path = upscaled_path
                upscale_info = {"enabled": True, "method": "EDSR_x4"}
            except Exception as e:
                print(f"Error en upscaling: {e}")
                upscale_info = {"enabled": False, "error": str(e)}
        
        # PASO 3: Preprocesamiento avanzado
        preprocessing_info = {"enabled": False}
        if enable_preprocessing:
            try:
                preprocessed_path = preprocess_for_ocr(
                    image_path,
                    enable_clahe=True,
                    enable_denoise=True,
                    enable_sharpen=True,
                    clahe_clip=2.5,
                    denoise_strength=10,
                    sharpen_amount=1.5
                )
                image_path = preprocessed_path
                preprocessing_info = {"enabled": True, "techniques": ["CLAHE", "Denoise", "Sharpen"]}
            except Exception as e:
                print(f"Error en preprocesamiento: {e}")
                preprocessing_info = {"enabled": False, "error": str(e)}
        
        # PASO 4: Extraer texto con OCR
        ocr_result = extract_text_from_image(image_path)
        
        if not ocr_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"OCR error: {ocr_result.get('error', 'unknown')}"
            )
        
        # PASO 5: Extraer campos con LLM
        try:
            formatted = extract_with_ollama(ocr_result.get("text_full", ""))  # ← CAMBIADO
        except Exception as e:
            print(f"Error en extracción LLM: {e}")
            formatted = {"success": False, "error": str(e)}
        
        return {
            "filename": name,
            "texto_extraido": ocr_result.get("text_full", ""),
            "ticket": formatted,
            "processing_info": {
                "crop_info": crop_info,
                "upscale_info": upscale_info,
                "preprocessing_info": preprocessing_info,
                "total_lines": ocr_result.get("total_lines", 0)
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/adquirir-base64/")
async def adquirir_base64(
    image: Base64ImageRequest,
    auto_crop: bool = Query(True, description="Detectar y recortar ticket automáticamente"),
    enable_upscaling: bool = Query(False, description="Super-resolution x4 (lento, solo para imágenes pequeñas)"),
    enable_preprocessing: bool = Query(True, description="Aplicar preprocesamiento CLAHE/Denoise/Sharpen")
):
    """
    Pipeline completo de procesamiento de tickets (base64):
    1. Segmentar ticket (auto_crop)
    2. Super-resolution x4 (enable_upscaling) - OPCIONAL
    3. Preprocesamiento CLAHE/Denoise/Sharpen (enable_preprocessing)
    4. OCR con EasyOCR
    5. Extracción de campos con LLM (Ollama)
    """
    try:
        result = process_base64_image(image.data)
        filename = result.get("filename")
        path = result.get("path")
        
        if not filename or not path:
            raise ValueError("Error al guardar la imagen base64")
        
        image_path = path
        
        # PASO 1: Detectar y recortar ticket
        crop_info = {"enabled": False}
        if auto_crop:
            try:
                cropped_path, crop_info = crop_ticket_from_image(image_path, save_debug=False)
                print(f"Ticket detectado: {crop_info.get('method')}")
                image_path = cropped_path
                crop_info["enabled"] = True
            except Exception as e:
                print(f"Error al recortar: {e}")
                crop_info = {"enabled": False, "error": str(e)}
        
        # PASO 2: Super-resolution x4 (OPCIONAL - lento)
        upscale_info = {"enabled": False}
        if enable_upscaling:
            try:
                upscaled_path = upscale_image_x4(image_path)
                image_path = upscaled_path
                upscale_info = {"enabled": True, "method": "EDSR_x4"}
            except Exception as e:
                print(f"Error en upscaling: {e}")
                upscale_info = {"enabled": False, "error": str(e)}
        
        # PASO 3: Preprocesamiento avanzado
        preprocessing_info = {"enabled": False}
        if enable_preprocessing:
            try:
                preprocessed_path = preprocess_for_ocr(
                    image_path,
                    enable_clahe=True,
                    enable_denoise=True,
                    enable_sharpen=True,
                    clahe_clip=2.5,
                    denoise_strength=10,
                    sharpen_amount=1.5
                )
                image_path = preprocessed_path
                preprocessing_info = {"enabled": True, "techniques": ["CLAHE", "Denoise", "Sharpen"]}
            except Exception as e:
                print(f"Error en preprocesamiento: {e}")
                preprocessing_info = {"enabled": False, "error": str(e)}
        
        # PASO 4: Extraer texto con OCR
        ocr_result = extract_text_from_image(image_path)
        
        if not ocr_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"OCR error: {ocr_result.get('error', 'unknown')}"
            )
        
        # PASO 5: Extraer campos con LLM
        try:
            formatted = extract_with_ollama(ocr_result.get("text_full", ""))  # ← CAMBIADO
        except Exception as e:
            print(f"Error en extracción LLM: {e}")
            formatted = {"success": False, "error": str(e)}
        
        return {
            "filename": filename,
            "path": image_path,
            "texto_extraido": ocr_result.get("text_full", ""),
            "ticket": formatted,
            "processing_info": {
                "crop_info": crop_info,
                "upscale_info": upscale_info,
                "preprocessing_info": preprocessing_info,
                "total_lines": ocr_result.get("total_lines", 0)
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/regresar-imagenes/", response_model=ImagesListResponse)
async def regresar_imagenes():
    """Retorna todas las imágenes procesadas en base64 (sin OCR)"""
    items = get_all_images()
    return {"images": items}