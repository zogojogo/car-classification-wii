import torch
from torchvision import transforms
import torchvision as vision
from PIL import Image
import io

def preprocess_img(img_path, mode):
    # transfor img to tensor
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if mode == 'api':
        img = Image.open(io.BytesIO(img_path))
    else:
        img = Image.open(img_path)

    test_transforms = transforms.Compose([
    vision.transforms.Resize((224, 224)),
    vision.transforms.ToTensor(),
    vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = test_transforms(img).to(device)
    img = img.unsqueeze(0)
    return img