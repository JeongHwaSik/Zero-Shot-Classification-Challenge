import torch.nn as nn

class Linear_Adapter(nn.Module):
    def __init__(self, c_in):
        super(Linear_Adapter, self).__init__()
        self.fc = nn.Linear(c_in, c_in)

    def forward(self, x):
        x = self.fc(x)
        return x