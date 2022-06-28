from src.preprocess import preprocess_img
from torch.nn import functional as F
import json
import matplotlib.pyplot as plt
from PIL import Image

def forward_pass(img_path, model, mode, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    img = preprocess_img(img_path, mode)
    model.eval()
    output = model(img)
    conf, predicted_id = F.softmax(output, dim=1).topk(topk)
    conf = conf.cpu().detach().numpy()[0]
    predicted_id = predicted_id.cpu().detach().numpy()[0]
    classes = [idx_to_classes(x) for x in predicted_id]
    return conf, predicted_id, classes

def idx_to_classes(idx):
    with(open('./classes.json', 'r')) as f:
        classes = json.load(f)
    return classes[idx]

def plot_prediction(img_path, model, topk=5):
    probs, id, classes = forward_pass(img_path, model, 'file', topk)
    plt.figure(figsize=(10, 10))
    plt.imshow(Image.open(img_path))
    plt.axis('off')
    plt.title('Prediction: {}, with Confidence: {}'.format(classes[0], probs[0]))
    plt.show()