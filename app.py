import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

 
st.title("Görüntü Sınıflandırma Uygulaması")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
model = models.resnet18(pretrained=True)
num_classes = 87   
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(torch.load("model_checkpoint_rest_net_ge.pth", map_location=device, weights_only=True))))
model.eval()   

 
transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
])

 
uploaded_file = st.file_uploader("Lütfen bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Resim', use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)


    import json
    with open("class_mapping.json", "r") as json_file:
        class_mapping = json.load(json_file)

    predicted_label = predicted.item()   
    predicted_class_name = class_mapping.get(str(predicted_label), "Bilinmeyen Sınıf")

    st.write(f"Tahmin Edilen Sınıf: {predicted_class_name}")
