import torch.nn as nn

class CLIP_Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(CLIP_Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
