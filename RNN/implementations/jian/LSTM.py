import torch
import torch.nn as nn

class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(simpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. The LSTM Layer
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 2. The Linear Layer (Standard Neural Network part)
        # This takes the LSTM output and maps it to our final prediction classes
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Initialize hidden state (h0) and cell state (c0)
        # If you don't do this, PyTorch defaults them to zeros,
        # but it's good practice to be explicit.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out: tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
        # _ : we ignore the final hidden/cell state tuple for this simple example
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # SHAPE CHECK:
        # 'out' shape is (batch_size, sequence_length, hidden_size)

        # DECODING:
        # We usually want the output from the LAST time step to make a prediction
        # out[:, -1, :] selects the last time step for the whole batch
        last_time_step_out = out[:, -1, :]

        # Pass through the linear layer
        prediction = self.fc(last_time_step_out)

        return prediction

# --- CONCRETE EXAMPLE ---

# Hyperparameters
INPUT_SIZE = 10     # e.g., size of word embedding vector
HIDDEN_SIZE = 20    # capacity of the LSTM memory
NUM_LAYERS = 2      # stacked LSTMs
OUTPUT_SIZE = 1     # e.g., 1 for binary classification (0 or 1)
BATCH_SIZE = 3      # number of sentences
SEQ_LENGTH = 5      # words per sentence

# Initialize Model
model = simpleLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# Create Dummy Input Data (Random numbers)
# Shape: (3 sentences, 5 words each, 10 features per word)
dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)

# Forward Pass
output = model(dummy_input)

print(f"Input Shape:  {dummy_input.shape}")
print(f"Output Shape: {output.shape}")
# Expected Output Shape: torch.Size([3, 1]) -> 1 prediction per sentence
