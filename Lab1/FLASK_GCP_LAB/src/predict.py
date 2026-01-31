import os
import io
import torch
from PIL import Image
from cnn import cnn
from train import run_training
import torchvision.transforms as transforms

# Load the trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "model", "model.pth")

model = torch.load(model_path, weights_only=False)

class_dict = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

def predict_class(img):
    image_bytes = img.read()

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    tensor = transform(image).unsqueeze(0) # handle batch dimension

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)

    return class_dict[predicted.item()]

if __name__ == "__main__":
    if os.path.exists("model/model.pth"):
        print("Model loaded successfully")
    else:
        model_dir = os.path.join(script_dir, "..", "model")
        os.makedirs(model_dir, exist_ok=True)
        run_training()
