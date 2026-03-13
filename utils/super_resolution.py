# utils/super_resolution.py
import cv2
import numpy as np
from typing import Optional
import os
import urllib.request


class SuperResolutionUpscaler:
    """
    Upscaler usando modelos EDSR y ESPCN de OpenCV
    Más ligero que Real-ESRGAN, no requiere PyTorch
    """
    
    def __init__(self, scale: int = 4, model_type: str = 'EDSR'):
        """
        Args:
            scale: Factor de escala (2, 3, o 4)
            model_type: 'EDSR' (mejor calidad) o 'ESPCN' (más rápido)
        """
        self.scale = scale
        self.model_type = model_type
        self.sr = None
        
    def load_model(self):
        """Carga el modelo de super-resolution"""
        if self.sr is not None:
            return
        
        print(f"Cargando modelo {self.model_type} x{self.scale}...")
        
        models_dir = os.path.join(os.getcwd(), 'models', 'opencv_sr')
        os.makedirs(models_dir, exist_ok=True)
        
        if self.model_type == 'EDSR':
            model_name = f'EDSR_x{self.scale}.pb'
            model_url = f'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/{model_name}'
        else:  # ESPCN
            model_name = f'ESPCN_x{self.scale}.pb'
            model_url = f'https://github.com/fannymonori/TF-ESPCN/raw/master/export/{model_name}'
        
        model_path = os.path.join(models_dir, model_name)
        
        # Descargar modelo si no existe
        if not os.path.exists(model_path):
            print(f"Descargando {model_name}...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"Modelo descargado: {model_path}")
            except Exception as e:
                raise Exception(f"Error descargando modelo: {e}")
        
        # Cargar modelo
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel(self.model_type.lower(), self.scale)
            print(f"Modelo cargado exitosamente")
        except Exception as e:
            raise Exception(f"Error cargando modelo: {e}")
    
    def upscale_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Aumenta la resolución de una imagen
        
        Args:
            image_path: Ruta de imagen de baja resolución
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta de la imagen upscaled
        """
        # Cargar modelo
        self.load_model()
        
        print(f"Procesando imagen con {self.model_type} x{self.scale}...")
        
        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo leer: {image_path}")
        
        original_size = f"{image.shape[1]}x{image.shape[0]}"
        print(f"  Tamaño original: {original_size}")
        
        # Aplicar super-resolution
        try:
            upscaled = self.sr.upsample(image)
        except Exception as e:
            print(f"Error en upscaling: {e}")
            raise
        
        upscaled_size = f"{upscaled.shape[1]}x{upscaled.shape[0]}"
        print(f"  Tamaño upscaled: {upscaled_size}")
        
        # Guardar resultado
        if output_path is None:
            output_path = image_path.replace('.png', '_upscaled.png').replace('.jpg', '_upscaled.jpg')
        
        cv2.imwrite(output_path, upscaled)
        print(f"Super-resolution completada: {output_path}")
        
        return output_path


# Funciones helper
def upscale_image_x4(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Upscale x4 usando EDSR (mejor calidad)
    
    Args:
        image_path: Ruta de imagen a mejorar
        output_path: Ruta de salida (opcional)
        
    Returns:
        Ruta de imagen upscaled
    """
    upscaler = SuperResolutionUpscaler(scale=4, model_type='EDSR')
    return upscaler.upscale_image(image_path, output_path)


def upscale_image_x2(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Upscale x2 (más rápido)
    
    Args:
        image_path: Ruta de imagen a mejorar
        output_path: Ruta de salida (opcional)
        
    Returns:
        Ruta de imagen upscaled
    """
    upscaler = SuperResolutionUpscaler(scale=2, model_type='EDSR')
    return upscaler.upscale_image(image_path, output_path)


def upscale_image_fast(image_path: str, output_path: Optional[str] = None, scale: int = 4) -> str:
    """
    Upscale rápido usando ESPCN (sacrifica un poco de calidad por velocidad)
    
    Args:
        image_path: Ruta de imagen a mejorar
        output_path: Ruta de salida (opcional)
        scale: Factor de escala (2, 3, o 4)
        
    Returns:
        Ruta de imagen upscaled
    """
    upscaler = SuperResolutionUpscaler(scale=scale, model_type='ESPCN')
    return upscaler.upscale_image(image_path, output_path)