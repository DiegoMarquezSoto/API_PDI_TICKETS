from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# SCHEMAS ORIGINALES
class Base64ImageRequest(BaseModel):
    data: str

class ImageResponse(BaseModel):
    name: str
    data: str

class ImagesListResponse(BaseModel):
    images: list[ImageResponse]

class Base64SaveResponse(BaseModel):
    status: bool
    message: str
    filename: str
    path: str
    size: str

class OCRLineResult(BaseModel):
    text: str
    confidence: float
    bbox: List

class OCRResponse(BaseModel):
    success: bool
    text_full: str
    lines: List[Dict]
    total_lines: int
    error:  Optional[str] = None

class OCRRequest(BaseModel):
    image_path: str

# ============ NUEVOS SCHEMAS PARA TICKETS ESTRUCTURADOS ============

class TicketAmounts(BaseModel):
    """Montos del ticket"""
    subtotal: Optional[str] = Field(None, description="Subtotal sin IVA")
    iva: Optional[str] = Field(None, description="IVA desglosado")
    propina: Optional[str] = Field(None, description="Propina si existe")
    total: Optional[str] = Field(None, description="Total a pagar")
    
class TicketMetadata(BaseModel):
    """Metadatos del ticket"""
    fecha:  Optional[str] = Field(None, description="Fecha de emisión")
    hora: Optional[str] = Field(None, description="Hora de emisión")
    referencia: Optional[str] = Field(None, description="Número de referencia/folio")
    autorizacion: Optional[str] = Field(None, description="Código de autorización")
    serie: Optional[str] = Field(None, description="Serie del documento FEL")
    dte: Optional[str] = Field(None, description="Número de autorización DTE")
    
class TicketInvoiceInfo(BaseModel):
    """Información de facturación"""
    url_factura: Optional[str] = Field(None, description="URL para facturar")
    tiene_facturacion: bool = Field(False, description="¿Tiene opción de factura?")
    
class TicketStructured(BaseModel):
    """Modelo completo de ticket estructurado"""
    establecimiento: Optional[str] = Field(None, description="Nombre del comercio")
    rfc: Optional[str] = Field(None, description="RFC del establecimiento")
    
    # Montos
    montos: TicketAmounts
    
    # Metadatos
    metadata: TicketMetadata
    
    # Facturación
    facturacion: TicketInvoiceInfo
    
    # Información adicional
    tiene_propina: bool = Field(False, description="¿Incluye propina?")
    metodo_pago: Optional[str] = Field(None, description="EFECTIVO/TARJETA/etc")
    
    # OCR raw
    texto_completo: str = Field("", description="Texto OCR completo")
    confianza_promedio: float = Field(0.0, description="Confianza promedio del OCR")

    #rfc: Optional[str] = Field(None, description="RFC del establecimiento")
    nit: Optional[str] = Field(None, description="NIT del establecimiento (Guatemala)")

class TicketProcessResponse(BaseModel):
    """Respuesta del endpoint de procesamiento"""
    filename: str
    ticket_estructurado: TicketStructured
    ocr_raw: Optional[Dict] = Field(None, description="Datos crudos del OCR (debug)")