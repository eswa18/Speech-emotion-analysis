"""
CNN-BiLSTM with Attention for Speech Emotion Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Simple additive attention over LSTM outputs."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = self.attention(lstm_output)       # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)    # (batch, seq_len, 1)
        context = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden_size)
        return context, attn_weights


class EmotionCNN_BiLSTM(nn.Module):
    """
    CNN-BiLSTM with Attention for speech emotion recognition.

    Input shape: (batch, feature_dim, time_steps)
      - feature_dim = 141 (combined features) or 128 (mel-spec)
      - time_steps = 200 (fixed_length)
    """

    def __init__(self, num_classes=4, input_size=141):
        super().__init__()

        # ── CNN Block: extract local patterns ──
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.3)

        # ── BiLSTM Block: capture temporal dependencies ──
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # ── Attention Block ──
        self.attention = Attention(hidden_size=256)  # 128 * 2 (bidirectional)

        # ── Classifier ──
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, feature_dim, time_steps)

        # CNN
        x = F.relu(self.bn1(self.conv1(x)))   # (batch, 64, T)
        x = self.pool(x)                       # (batch, 64, T/2)
        x = F.relu(self.bn2(self.conv2(x)))   # (batch, 128, T/2)
        x = self.pool(x)                       # (batch, 128, T/4)
        x = F.relu(self.bn3(self.conv3(x)))   # (batch, 256, T/4)
        x = self.dropout_cnn(x)

        # Reshape for LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)                 # (batch, T/4, 256)

        # BiLSTM
        lstm_out, _ = self.lstm(x)             # (batch, T/4, 256)

        # Attention pooling
        context, _ = self.attention(lstm_out)   # (batch, 256)

        # Classifier
        x = F.relu(self.fc1(context))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


def get_model(num_classes=4, input_size=141):
    """Factory function to create the model."""
    return EmotionCNN_BiLSTM(num_classes=num_classes, input_size=input_size)
