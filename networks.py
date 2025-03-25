from typing import Any, Dict
import torch.nn as nn


""" Define the neural network architecture, used as selector or predictor. """

class DeepSet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Add layers with proper initialization
        for _ in range(num_layers):
            linear = nn.Linear(prev_dim, hidden_dim)
            # Initialize weights properly
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            
            layers.extend([
                linear,
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),  # Add BatchNorm
                nn.Dropout(0.2)  # Add dropout
            ])
            prev_dim = hidden_dim
            
        # Final layer
        self.final = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return self.final(x)


class StackLinear(nn.Module):
    """ This is the same as Deepset, make this change only for LassoNet since it's easier to compute weights norm."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(StackLinear, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim, self.num_layers = input_dim, hidden_dim, output_dim, num_layers
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for l in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))    
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(self.input_dim, momentum=0.01)

    def forward(self, Input):
        current_layer = self.batchnorm(Input)
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = self.tanh(current_layer)
        return current_layer








