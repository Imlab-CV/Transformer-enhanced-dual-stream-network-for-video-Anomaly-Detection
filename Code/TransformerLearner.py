
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim=2048, num_heads=8, hidden_dim=512, num_layers=3, dropout=0.0):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print(x.shape)
        x=x.reshape(x.shape[0],1,x.shape[1])
        #print(x.shape)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change from (batch_size, seq_len, hidden_dim) to (seq_len, batch_size, hidden_dim)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x.mean(dim=0)  # Average across sequence length
        x = self.fc(x)
        return torch.sigmoid(x)

class TransformerLearner(nn.Module):
    def __init__(self, input_dim=2048, num_heads=8, hidden_dim=512, num_layers=3, dropout=0.0):
        super(TransformerLearner, self).__init__()
        self.transformer = Transformer(input_dim, num_heads, hidden_dim, num_layers, dropout)

    def forward(self, x):
        return self.transformer(x)


