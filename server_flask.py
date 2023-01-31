import requests
import torch
from pathlib import Path
from PIL import Image
from io import BytesIO
from flask import Flask
from flask import request
from server_model import BestModel

app = Flask(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BestModel(
    models_path=Path("./models/3/MobileNetV3-10.pt"),
    device=device,
)


@app.route("/predict", methods=['GET'])
def predict():
    url = request.args.get('url')

    with requests.get(url) as r:
        img = Image.open(BytesIO(r.content))

    if request.method == 'GET':
        return {
            'url': url,
            'prediction': model.predict(img)
        }
    else:
        return {
            'url': 0,
            'prediction': "Please use GET for prediction"
        }
