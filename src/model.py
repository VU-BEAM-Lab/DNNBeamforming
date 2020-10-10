# Copyright 2020 Jaime Tierney, Adam Luchies, and Brett Byram

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the license at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

from torch import nn
import torch
import os



class FullyConnectedNet(nn.Module):
    """Fully connected network. ReLU is the activation function.
        Network parameters are intialized with a normal distribution.

    Args:
        input_dim
        output_dim
        layer_width
        num_hidden
        dropout
        dropout_input
        starting_weights
        batch_norm_enable

    """
    def __init__(self, input_dim,
                        output_dim,
                        layer_width,
                        num_hidden=1,
                        dropout=0,
                        dropout_input=0,
                        starting_weights=None,
                        batch_norm_enable=False):

        super().__init__()

        # input connects to first hidden layer
        self.layers = nn.ModuleList([nn.Linear(input_dim, layer_width)])

        # hidden layers
        for i in range(num_hidden-1):
            self.layers.append(nn.Linear(layer_width, layer_width))

        # last hidden connects to output layer
        self.layers.append(nn.Linear(layer_width, output_dim))

        # activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)

        # batch norm layers
        self.batch_norm_enable = batch_norm_enable
        if batch_norm_enable:
            self.bn_layers = nn.ModuleList([])
            for i in range(num_hidden):
                self.bn_layers.append(nn.BatchNorm1d(layer_width))

        # initialize weights
        self.starting_weights = starting_weights
        try:
            self.load_state_dict(torch.load(starting_weights))
            print('Using weights: ' + str(self.starting_weights))
        except:
            print('Initializing network weights randomly.')
            self._initialize_weights()

    def forward(self, x):

        # input dropout
        x = self.dropout_input(x)

        for i in range( len(self.layers) -1 ):
            x = self.layers[i](x)
            if self.batch_norm_enable:
                x = self.bn_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x)

        # no dropout or activtion function on the last layer
        x = self.layers[-1](x)

        return x

    def _initialize_weights(self):

        for i in range(len(self.layers)):
            nn.init.kaiming_normal_( self.layers[i].weight.data )
            self.layers[i].bias.data.fill_(0.01)
