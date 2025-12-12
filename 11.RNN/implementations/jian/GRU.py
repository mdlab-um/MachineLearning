import torch
import torch.nn as nn

class simpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(simpleGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. The GRU Layer
        # Notice we simply swap nn.LSTM for nn.GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 2. The Linear Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Initialize hidden state (h0)
        # CRITICAL DIFFERENCE: GRU has NO cell state (c0). Only h0.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        # out: tensor containing the output features (h_t) from the last layer of the GRU, for each t.
        # hn : tensor containing the hidden state for the last time step.
        out, hn = self.gru(x, h0)

        # DECODING:
        # Just like LSTM, we usually take the output from the last time step
        last_time_step_out = out[:, -1, :]

        # Pass through the linear layer
        prediction = self.fc(last_time_step_out)

        return prediction

# --- CONCRETE EXAMPLE ---

# Hyperparameters
INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 3
SEQ_LENGTH = 5

# Initialize Model
model = simpleGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# Create Dummy Input Data
dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)

# Forward Pass
output = model(dummy_input)

print(f"Input Shape:  {dummy_input.shape}")
print(f"Output Shape: {output.shape}")
