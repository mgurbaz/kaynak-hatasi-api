from flask import Flask, request, render_template_string, jsonify
from roboflow import Roboflow
import cv2
import numpy as np
import os

app = Flask(__name__)

rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", ""))
project = rf.workspace("taha-antzq").project("kaynak-hatalari")
version = project.version(1)
model = version.model

HTML = """
<!doctype html>
<title>Kaynak Hatası Test</title>
<h2>Kaynak Hatası Görseli Yükle</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=image required>
  <input type=submit value=Yükle>
</form>
{% if prediction %}
<h3>Sonuçlar:</h3>
<pre>{{ prediction }}</pre>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return "Dosya bulunamadı", 400
        image = file.read()
        npimg = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        img_path = "/tmp/input.jpg"
        cv2.imwrite(img_path, img)

        prediction = model.predict(img_path, confidence=40, overlap=30).json()
    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
