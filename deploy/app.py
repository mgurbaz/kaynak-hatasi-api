from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io

app = FastAPI()

# Modeli yükle (model.pt aynı klasörde olacak)
model = torch.hub.load('ultralytics/yolov8', 'custom', path='model.pt', force_reload=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = model(img)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return JSONResponse(content={"detections": detections})
