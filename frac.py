import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the MNIST dataset
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = load_data()

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Expanded and more granular grid search parameters
learning_rates = torch.logspace(-4, 0, 20)  # Logarithmic scale from 1e-4 to 1
hidden_units = torch.linspace(10, 200, 20, dtype=torch.int)  # Linear space from 10 to 200
epochs = 5

# Placeholder for results
results = torch.zeros(len(learning_rates), len(hidden_units))

# Grid search
for i, lr in enumerate(learning_rates):
    for j, hu in enumerate(hidden_units):
        model = SimpleNN(hu.item()).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the model
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the training success
        success = loss.item() < 0.5  # Example criterion
        results[i, j] = success

# Visualize the results
import matplotlib.pyplot as plt
plt.imshow(results, cmap='viridis', extent=[hidden_units.min(), hidden_units.max(), learning_rates.min(), learning_rates.max()])
plt.colorbar(label='Success (1) or Failure (0)')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Learning Rate (log scale)')
plt.title('Training Success by Hyperparameters in PyTorch')
plt.yscale('log')
plt.show()
