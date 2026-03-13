from fastapi import FastAPI
from routers.image_routes import router as image_router

app = FastAPI(
    title="API PDI - Procesamiento Digital de Imágenes con OCR",
    description="API para procesar imágenes y extraer texto",
    version="1.0.0"
)

### MIS RUTAS ###
app.include_router(image_router)

@app.get("/")
def read_root():
    return {"message": "API PDI con OCR funcionando correctamente"}