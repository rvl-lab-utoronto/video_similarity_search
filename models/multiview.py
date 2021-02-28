import torch
import torch.nn as nn
import torch.nn.functional as F

class Multiview(nn.Module):
    def __init__(self, encoder1, encoder2, embed_dim=128):
        super(Multiview, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

        self.combine = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(2*embed_dim), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*embed_dim, 2 * embed_dim),
                    torch.nn.BatchNorm1d(2*embed_dim), 
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1), 
                    torch.nn.Linear(2*embed_dim, embed_dim))

    def forward(self, views):
        # print('forwarding...')
        (view1, view2) = views
        embedded_1 = self.encoder1(view1)
        embedded_2 = self.encoder2(view2)
        embedded = torch.cat((embedded_1, embedded_2), dim=1)
        fused_embedded = self.combine(embedded)
        return fused_embedded

