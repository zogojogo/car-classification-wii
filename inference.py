from src.load_model import load_model
from src.predict import forward_pass, plot_prediction
from models.network import Net
    
if __name__ == "__main__":
    model = load_model('models/checkpoint_74.27')
    img_path = 'test/veyron_1.jpg'
    probs, id, classes = forward_pass(img_path, model, mode='x', topk=5)
    plot_prediction(img_path, model)
    print(probs)
    print(id)
    print(classes)
