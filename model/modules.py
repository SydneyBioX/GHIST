import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, n_classes, n_genes, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query_transform = nn.Linear(n_classes, embed_dim)
        self.key_transform = nn.Linear(n_genes, embed_dim)
        self.value_transform = nn.Linear(n_genes, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        self.linear_out = nn.Linear(embed_dim, n_genes)

    def forward(self, comp, expr):
        # Input tensor of shape (batch_size, n_classes/n_genes)
        comp = comp.unsqueeze(0)
        expr = expr.unsqueeze(0)

        query = self.query_transform(comp)
        key = self.key_transform(expr)
        value = self.value_transform(expr)

        output, _ = self.attention(query, key, value)
        output = output.squeeze(0)
        output = self.linear_out(output)

        return output


class Embed(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Embed, self).__init__()

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, hidden_size, num_classes):
        super(MLP, self).__init__()

        layers_mlp = [
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        ]
        self.mlp = nn.Sequential(*layers_mlp)

    def forward(self, x):
        out = self.mlp(x)
        fv = self.mlp[0](x)
        return out, fv


class MLPSoftmax(torch.nn.Module):
    def __init__(self, in_size, hidden_size, num_classes):
        super(MLPSoftmax, self).__init__()

        layers_mlp = [
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        ]
        self.mlp = nn.Sequential(*layers_mlp)

    def forward(self, x):
        out = self.mlp(x)
        return out
