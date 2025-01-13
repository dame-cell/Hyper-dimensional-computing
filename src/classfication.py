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

# Add argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='HD Computing MNIST Classification')
    parser.add_argument('--dimensions', type=int, default=10000,
                        help='number of dimensions (default: 10000)')
    parser.add_argument('--img-size', type=int, default=28,
                        help='input image size (default: 28)')
    parser.add_argument('--num-levels', type=int, default=1000,
                        help='number of levels (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument('--data-dir', type=str, default="../data",
                        help='data directory (default: ../data)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    transform = torchvision.transforms.ToTensor()

    train_ds = MNIST(args.data_dir, train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    test_ds = MNIST(args.data_dir, train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    class Encoder(nn.Module):
        def __init__(self, out_features, size, levels):
            super(Encoder, self).__init__()
            self.flatten = torch.nn.Flatten()
            self.position = embeddings.Random(size * size, out_features)
            self.value = embeddings.Level(levels, out_features)

        def forward(self, x):
            x = self.flatten(x)
            sample_hv = torchhd.bind(self.position.weight, self.value(x))
            sample_hv = torchhd.multiset(sample_hv)
            return torchhd.hard_quantize(sample_hv)

    encode = Encoder(args.dimensions, args.img_size, args.num_levels)
    encode = encode.to(device)

    num_classes = len(train_ds.classes)
    model = Centroid(args.dimensions, num_classes)
    model = model.to(device)

    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples)
            model.add(samples_hv, labels)

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():
        model.normalize()
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), labels)

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")

if __name__ == "__main__":
    main()