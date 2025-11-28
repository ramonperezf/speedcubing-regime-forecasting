import torch
import torch.nn as nn

class HybridLSTM(nn.Module):
    """
    Standard LSTM Architecture optimized for small datasets.
    Uses a single layer to prevent overfitting while capturing temporal dependencies.
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(HybridLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        
        out = hn.squeeze(0)
        
        out = self.dropout(out)
        prediction = self.fc(out)
        return prediction