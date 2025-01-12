from torch import nn

class FCModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input_data):
        return self.net(input_data)