import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
import os
import json

MODEL_PATH = r"C:\Users\Shruthilaya\GUVI\covid_classifier_resnet18.pth"   
CLASSES_PATH = r"C:\Users\Shruthilaya\GUVI\classes.json"                
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Helpers ----------
def load_class_names(path=CLASSES_PATH):
    """
    Try to load class names from a json file. If not found, use a sensible default order.
    IMPORTANT: If your class ordering is different, replace the list below with your dataset.classes order.
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                classes = json.load(f)
            st.info(f"Loaded class names from {path}")
            return classes
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    # default fallback - change if your training used a different order
    st.warning("classes.json not found — using default class order. "
               "If predictions look wrong, create a classes.json with the correct order.")
    return ["COVID-19", "Normal", "ViralPneumonia"]

def build_model(num_classes):
    """
    Build a ResNet18 with a final linear layer sized to num_classes.
    This must match the architecture you used during training.
    """
    model = models.resnet18(pretrained=False)  # pretrained=False since we'll load weights
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# Preprocessing for inference (no augmentation)
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

@st.cache_resource
def load_model(model_path, num_classes):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image: Image.Image, transform, class_names):
    """
    Returns (predicted_label, probs_array)
    probs_array is ordered according to class_names
    """
    img_t = transform(image).unsqueeze(0).to(DEVICE)  # shape [1,3,H,W]
    with torch.no_grad():
        logits = model(img_t)            # raw scores
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        top_idx = int(np.argmax(probs))
        return class_names[top_idx], probs

# ---------- Streamlit UI ----------
st.set_page_config(page_title="X-ray Multi-class Detector", layout="centered")
st.title("Chest X-ray Multi-class Detector (Demo)")
st.caption("Demo — not medical advice. Use only for learning/testing.")

st.write("Upload a chest X-ray image. Supported types: jpg, png, jpeg.")

# load classes (or fallback) and model (lazy load)
class_names = load_class_names()
num_classes = len(class_names)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Train the model and place the .pth file there.")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH, num_classes)
transform = get_transform()

uploaded = st.file_uploader("Choose an X-ray image", type=["jpg","jpeg","png"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Couldn't open the image: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Run Inference"):
        with st.spinner("Predicting..."):
            pred_label, probs = predict_image(model, image, transform, class_names)
        st.success(f"Predicted: **{pred_label}**")

        # Show probabilities
        st.subheader("Class probabilities")
        for cname, p in zip(class_names, probs):
            st.write(f"- {cname}: {p*100:.2f}%")

        # Top 3
        topk = min(3, len(class_names))
        top_idxs = np.argsort(probs)[::-1][:topk]
        st.subheader(f"Top {topk}")
        for i in top_idxs:
            st.write(f"{i+1}. {class_names[i]} — {probs[i]*100:.2f}%")

