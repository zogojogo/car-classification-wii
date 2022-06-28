import torch

def load_model(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model.to(device)
    return model