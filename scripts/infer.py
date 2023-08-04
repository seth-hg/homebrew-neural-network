from datetime import datetime
import torch
from torchvision import datasets, transforms

from mlp import Net

state_dict = torch.load("mnist_mlp.pt")

m = Net()
m.load_state_dict(state_dict)

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("../data", train=False, transform=transform)

test = dataset.data / 255
before = datetime.now()
m.forward(test)
elapsed = datetime.now() - before

elapsed_ms = elapsed.seconds * 1000.0 + elapsed.microseconds / 1000.0
print(f"Elapsed: {elapsed_ms} ms")
