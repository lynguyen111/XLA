import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from src.models.resnet import get_resnet18

app = Flask(__name__)
CORS(app)

CLASSES = [
    "Bo_canh_cung",
    "Bo_rua",
    "Buom",
    "Chau_chau",
    "Chuon_chuon",
    "Gian",
    "Kien",
    "Muoi",
    "Nhen",
    "Ong",
    "Ong_bap_cay",
    "Ruoi",
    "Sau_buom",
]

CLASS_DISPLAY = {
    "Kien": "Kiến",
    "Ong": "Ong",
    "Buom": "Bướm",
    "Chuon_chuon": "Chuồn Chuồn",
    "Chau_chau": "Châu Chấu",
    "Muoi": "Muỗi",
    "Gian": "Gián",
    "Bo_rua": "Bọ Rùa",
    "Bo_canh_cung": "Bọ Cánh Cứng",
    "Sau_buom": "Sâu Bướm",
    "Ruoi": "Ruồi",
    "Nhen": "Nhện",
    "Ong_bap_cay": "Ong Bắp Cày",
}

CLASS_EMOJI = {
    "Kien": "🐜",
    "Ong": "🐝",
    "Buom": "🦋",
    "Chuon_chuon": "🪲",
    "Chau_chau": "🦗",
    "Muoi": "🦟",
    "Gian": "🪳",
    "Bo_rua": "🐞",
    "Bo_canh_cung": "🪲",
    "Sau_buom": "🐛",
    "Ruoi": "🪰",
    "Nhen": "🕷️",
    "Ong_bap_cay": "🐝",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_saved"
)
models = {}


def load_models():
    print(f"Loading models on device: {DEVICE}")

    resnet = get_resnet18(num_classes=13, dropout=0.2)
    state = torch.load(
        "/Users/nguyenly/Desktop/code/XLA/models_saved/best_resnet18.pth",
        map_location=DEVICE,
    )
    resnet.load_state_dict(state)
    resnet.to(DEVICE)
    resnet.eval()
    models["resnet18"] = resnet
    print("ResNet18 loaded OK")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": list(models.keys()),
        "device": str(DEVICE),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Không có file ảnh được gửi lên"}), 400

    file = request.files["file"]
    model_name = request.form.get("model", "resnet18")

    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' chưa được load"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Ảnh không hợp lệ: {str(e)}"}), 400

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = models[model_name](tensor)
        probs = torch.softmax(output, dim=1)[0]

    top = torch.topk(probs, min(5, len(CLASSES)))

    predictions = []
    for i in range(top.indices.size(0)):
        idx = top.indices[i].item()
        conf = top.values[i].item()
        cls = CLASSES[idx]
        predictions.append({
            "class_id": idx,
            "class_name": cls,
            "display_name": CLASS_DISPLAY.get(cls, cls),
            "emoji": CLASS_EMOJI.get(cls, ""),
            "confidence": round(conf * 100, 2),
            "rank": i + 1,
        })

    return jsonify({
        "predictions": predictions,
        "model": model_name,
        "device": str(DEVICE),
    })


if __name__ == "__main__":
    load_models()
    app.run(debug=True, port=5001, host="0.0.0.0")
