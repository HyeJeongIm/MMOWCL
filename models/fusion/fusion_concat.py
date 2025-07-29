import torch
import torch.nn as nn
from torch.nn.init import normal_, constant_

class FusionConcat(nn.Module):
    def __init__(self, feature_dim, modality, dropout):
        super().__init__()
        self.modality = modality
        input_dim = len(modality) * feature_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # weight init
        normal_(self.fc1.weight, 0, 0.001)
        constant_(self.fc1.bias, 0)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        return {'features': x}
