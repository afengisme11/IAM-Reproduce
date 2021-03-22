'''
Author: your name
Date: 2021-03-18 11:51:56
LastEditTime: 2021-03-20 17:16:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /origin/home/zheyu/Desktop/Deep_Learning/reproduce/net_fnn.py
'''
import torch
import torch.nn as nn

class Net_fnn(nn.Module):
    """
    Build feedforward neural network.
    """
    def __init__(self, input, num_fc_layers, num_fc_units):
        """
        input: shape [batch_size, k]
        """
        super(Net_fnn, self).__init__()                    # Inherited from the parent class nn.Module
        self.layers = [nn.Linear(input.size()[1], num_fc_units[0]),nn.ReLU()]
        for i in range(num_fc_layers-1):
            self.layers.append(nn.Linear(num_fc_units[i], num_fc_units[i+1]))
            self.layers.append(nn.ReLU())
        self.layers = torch.nn.ModuleList(self.layers)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.layers[0](x)
        for j in range(1,len(self.layers)):
            out = self.layers[j](out)
        return out

class Net_cnn(nn.Module):
    """
    Build cnn.
    """
    def __init__(self, in_channels, n_filters, num_conv_layers, kernel_sizes, strides):
        """
        input: self.parameters["num_frames"]
        """
        super(Net_cnn, self).__init__()

        conv1 = nn.Conv2d(in_channels, n_filters[0],
                               kernel_size=kernel_sizes[0],
                               strides=strides[0],
                               padding=0)
        self.layers = [conv1,nn.ReLU()]                       
        for i in range(num_conv_layers-1):
            conv = nn.Conv2d(n_filters[i], n_filters[i+1],
                               kernel_size=kernel_sizes[i+1],
                               strides=strides[i+1],
                               padding=0) 
            self.layers.append(conv)  
            self.layers.append(nn.ReLU())
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self,x):
        out = self.layers[0](x)
        for j in range(1,len(self.layers)):
            out = self.layers[j](out)
        return out

# class Net_rnn(nn.Module):
#     """
#     Build LSTM rnn.
#     """


                         
