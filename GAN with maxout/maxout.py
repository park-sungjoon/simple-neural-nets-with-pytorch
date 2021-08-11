import torch
import torch.nn as nn

class Maxout(nn.Module):
    def __init__(self,in_features,out_features,num_pieces,irange=0.005,init_bias=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        self.fc_list = nn.ModuleList()
        self.irange = irange
        self.init_bias = init_bias
        for i in range(num_pieces):
            self.fc_list.append(nn.Linear(in_features,out_features,bias=True))
        self._init_weights()
    def _init_weights(self):
        for m in self.fc_list:
            nn.init.uniform_(m.weight.data,-self.irange,self.irange)
            if m.bias is not None:
                torch.zero_(m.bias.data)
     
    def forward(self,input_):
        z_list = []
        for layer in self.fc_list:
            z_list.append(layer(input_))
        return torch.max(torch.stack(z_list),dim=0)[0]