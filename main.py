from flask import Flask, request, jsonify
from roboflow import Roboflow
import cv2
import numpy as np
import os

app = Flask(__name__)

# Roboflow modelini initialize et
rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", ""))
project = rf.workspace("taha-antzq").project("kaynak-hatalari")
version = project.version(1)
model = version.model

@app.route('/')
def home():
    return "Kaynak Hatası Tespit API'si Çalışıyor"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image = file.read()
    npimg = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    cv2.imwrite("input.jpg", img)

    # Roboflow modelini kullanarak tahmin yap
    prediction = model.predict("input.jpg", confidence=40, overlap=30).json()

    return jsonify(prediction)
