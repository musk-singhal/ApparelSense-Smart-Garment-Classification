import json

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from PIL import Image
logging.basicConfig(filename='prediction_log.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def load_model(model_path, class_names):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_class(image_path, model, idx_to_class, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = idx_to_class[int(preds.item())]  # Convert index to class name

    parts = predicted_class.split()
    predicted_gender = parts[0]
    predicted_sleeve = parts[1]
    print(f"Predicted Gender: {predicted_gender}")
    print(f"Predicted Sleeve: {predicted_sleeve}")

    return predicted_gender, predicted_sleeve


def load_class_names(class_names_path):
    with open(class_names_path) as f:
        class_to_idx = json.load(f)
    # Invert the dictionary to map indices to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


def predict_label(
        image_url=r"/Users/muskan/Desktop/images/male_full_sleeve/image_1.jpg",
        model_save_path='ml_model/model_resnet18.pth'

):
    class_names_path = model_save_path.replace('.pth', '_class_names.json')
    image_path = image_url  # Change to your image path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_to_class = load_class_names(class_names_path)
    model = load_model(model_save_path, idx_to_class)
    model = model.to(device)

    predicted_gender, predicted_sleeve = predict_class(image_path, model, idx_to_class, device)

    return predicted_gender, predicted_sleeve
