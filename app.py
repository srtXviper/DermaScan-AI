import io, os
from functools import lru_cache
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

REPO = os.getenv("HF_REPO_ID", "SRTxVIPER/dermascan-skin-cnn")
WEIGHTS = os.getenv("MODEL_FILENAME", "skin_cnn.pth")
app = FastAPI(title="DermaScan AI - Educational Prototype")
app.mount("/static", StaticFiles(directory="static"), name="static")

@lru_cache(maxsize=1)
def load_model():
    path = hf_hub_download(repo_id=REPO, filename=WEIGHTS)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    names = ckpt["class_names"]
    n = int(ckpt["num_classes"])
    if len(names) != n:
        raise RuntimeError("Checkpoint class count does not match class_names.")
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, n)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()
    size = int(ckpt.get("image_size", 224))
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])
    prep = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return net, list(names), prep

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status":"ok", "app":"DermaScan AI", "model_repository":REPO,
            "model_file":WEIGHTS, "model_loaded":load_model.cache_info().currsize > 0}

@app.get("/api/model-info")
def model_info():
    try:
        _, names, _ = load_model()
        return {"available":True, "architecture":"ResNet18", "class_count":len(names), "classes":names}
    except Exception as e:
        print("MODEL LOAD ERROR:", repr(e), flush=True)
        raise HTTPException(503, "Model unavailable. Check the Hugging Face file and Render logs.")

@app.post("/api/image-check")
async def image_check(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg","image/png","image/webp"}:
        raise HTTPException(415, "Upload a JPG, PNG, or WEBP image.")
    data = await image.read(8*1024*1024+1)
    if not data: raise HTTPException(400, "The image is empty.")
    if len(data)>8*1024*1024: raise HTTPException(413, "Image must be 8 MB or smaller.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in {"JPEG","PNG","WEBP"}: raise HTTPException(415,"Unsupported image format.")
            w,h=im.size
            if w<64 or h<64: raise HTTPException(400,"Image must be at least 64 x 64 pixels.")
            rgb=im.convert("RGB")
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
        raise HTTPException(400,"The uploaded file is not a valid image.")
    try:
        net,names,prep=load_model()
        with torch.inference_mode():
            probs=torch.softmax(net(prep(rgb).unsqueeze(0)),dim=1)[0]
        vals,inds=torch.topk(probs,k=min(3,len(names)))
        preds=[{"label":str(names[int(i)]),"score":round(float(v),4)} for v,i in zip(vals,inds)]
    except Exception as e:
        print("INFERENCE ERROR:",repr(e),flush=True)
        raise HTTPException(503,"CNN model could not load or run. Check that the public Hugging Face repository contains skin_cnn.pth. See Render logs.")
    return {"accepted":True,"width":w,"height":h,"model":"ResNet18 CNN (experimental)",
            "predictions":preds,
            "disclaimer":"Educational prototype only. This model is not clinically validated. Scores are model outputs, not medical probabilities or a diagnosis. Do not use them to choose treatment; consult a qualified healthcare professional."}
