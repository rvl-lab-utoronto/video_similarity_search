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
        
        self.decoder1 = self.get_decoder(embed_dim)
        self.decoder2 = self.get_decoder(embed_dim)

    def get_decoder(self, embed_dim): #modified the output to be embed_dim
        # return torch.nn.Sequential(
        #             torch.nn.Linear(embed_dim, 2*embed_dim),
        #             torch.nn.Dropout(0.1),
        #             torch.nn.ReLU(),
        #             torch.nn.BatchNorm1d(2*embed_dim), 
        #             torch.nn.Linear(2*embed_dim, embed_dim),
        #             torch.nn.ReLU(),
        #             torch.nn.BatchNorm1d(embed_dim))

        return torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, embed_dim),
                    torch.nn.Dropout(0.1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(embed_dim), 
                    torch.nn.Linear(embed_dim, embed_dim),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(embed_dim))

    def forward(self, views):
        # print('forwarding...')
        (view1, view2) = views
        embedded_1 = self.encoder1(view1)
        embedded_2 = self.encoder2(view2)
        embedded = torch.cat((embedded_1, embedded_2), dim=1)
        fused_embedded = self.combine(embedded)

        decoded_1 = self.decoder1(fused_embedded)
        decoded_2 = self.decoder2(fused_embedded) #(batch_size, 128)
        return fused_embedded, (embedded_1, embedded_2), (decoded_1, decoded_2)


