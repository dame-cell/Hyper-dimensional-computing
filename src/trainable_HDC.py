# this is the example code from the original torchhd repository
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchmetrics
from tqdm import tqdm
import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Trainable HD Computing Classification')
    parser.add_argument('--dimensions', type=int, default=10000,
                        help='number of dimensions (default: 10000)')
    parser.add_argument('--img_size', type=int, default=28,
                        help='input image size (default: 28)')
    parser.add_argument('--num_levels', type=int, default=1000,
                        help='number of levels (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch size (default: 24)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='data directory (default: ./data)')
    parser.add_argument('--epochs', type=int, default='10',
                        help='how many iterations to train the model')
    return parser.parse_args()



def main():
    args = parse_args()
    # Data Preparation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    class LearnableCentroid(nn.Module):
        def __init__(self, dimensions, num_classes):
            super(LearnableCentroid, self).__init__()
            self.class_vectors = nn.Parameter(torch.randn(num_classes, dimensions))

        def forward(self, x):
            # Cosine similarity between input hypervector and class hypervectors
            similarities = F.cosine_similarity(x.unsqueeze(1), self.class_vectors.unsqueeze(0), dim=2)
            return similarities

    class Encoder(nn.Module):
        def __init__(self, out_features, size, levels):
            super(Encoder, self).__init__()
            self.flatten = torch.nn.Flatten()
            self.position = embeddings.Random(size * size, out_features,requires_grad =True)
            self.value = embeddings.Level(levels, out_features,requires_grad =True)

        def forward(self, x):
            x = self.flatten(x)
            sample_hv = torchhd.bind(self.position.weight, self.value(x))
            sample_hv = torchhd.multiset(sample_hv)
            return torchhd.hard_quantize(sample_hv)

    # Initialize model and encoder
    encode = Encoder(args.dimensions, args.img_size, args.num_levels).to(device)
    model = LearnableCentroid(args.dimensions, 10).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):  # Number of epochs
        model.train()
        total_loss = 0
        for samples, labels in tqdm(train_ld, desc=f"Training Epoch {epoch+1}"):
            samples, labels = samples.to(device), labels.to(device)

            # Encode samples
            samples_hv = encode(samples)

            # Forward pass
            outputs = model(samples_hv)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_ld)}")

        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for samples, labels in tqdm(test_ld, desc="Testing"):
                samples, labels = samples.to(device), labels.to(device)
                samples_hv = encode(samples)
                outputs = model(samples_hv)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()