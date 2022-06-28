import timm
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

class Net(nn.Module):
	
    def __init__(self):
      super(Net, self).__init__()
      self.num_classes = 196
      self.basemodel = timm.create_model('efficientnet_b0', pretrained=True)
      self.unfreeze_weight()
      self.filters = self.basemodel.classifier.in_features
      self.basemodel.classifier = nn.Sequential(
            # nn.Linear(self.filters, 128),
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(self.filters, self.num_classes),
            nn.LogSoftmax(dim=1)                    
        )

    def forward(self, x):
      x = self.basemodel(x)
      return x

    def freeze_weight(self):
      for param in self.basemodel.parameters():
            param.requires_grad = False # Freezing Weight
    
    def unfreeze_weight(self):
      for param in self.basemodel.parameters():
            param.requires_grad = True # Freezing Weight