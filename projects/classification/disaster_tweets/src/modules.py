from allennlp.common.from_params import FromParams
import torch.nn as nn
import torch
import math

class GELU(nn.Module):
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PosWiseFeedForward(nn.Module, FromParams):
    
    def __init__(self, d_model:int, 
                 d_ff:int, ):
        ''' Simple feed forward neural network '''
        super(PosWiseFeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model,bias=False)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)

