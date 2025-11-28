import torch
import torch.nn as nn

class HybridLSTM(nn.Module):
    """
    Standard LSTM Architecture optimized for small datasets.
    Uses a single layer to prevent overfitting while capturing temporal dependencies.
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(HybridLSTM, self).__init__()
        
        # LSTM Layer
        # batch_first=True means input is (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        
        # Dropout to prevent memorization
        self.dropout = nn.Dropout(0.2)
        
        # Regression Head
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # LSTM output: (batch, seq_len, hidden_dim)
        # We also get hidden state (h_n) and cell state (c_n)
        _, (hn, _) = self.lstm(x)
        
        # Use the final hidden state of the last layer
        # hn shape: (1, batch, hidden_dim) -> squeeze to (batch, hidden_dim)
        out = hn.squeeze(0)
        
        out = self.dropout(out)
        prediction = self.fc(out)
        return prediction