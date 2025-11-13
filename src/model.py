import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPatchNet(nn.Module):

#builds a simple 3-layer convolutional network to classify image patches
#categories like nodule, fauna, or background

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.fc(x)

def load_model(weights_path=None, num_classes=3, device=None):
#provides a convenient way to load pre-trained weights and prepare the model for inference
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPatchNet(num_classes=num_classes).to(device)
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        state = state.get("state_dict", state)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, device

