import argparse
import torch
from struct import pack


def save_model(state_dict, path):
    magic = pack("4s", b"HBNN")
    size = 0
    with open(path, "wb+") as f:
        size += f.write(magic)

        # write number of layers
        size += f.write(pack("<I", 2))

        # write weights & biases for layers
        for layer in ["hidden", "output"]:
            weight = state_dict[f"{layer}.weight"]
            rows, cols = weight.shape
            size += f.write(pack("<2I", rows, cols))
            size += f.write(weight.cpu().numpy().tobytes())
            if f"{layer}.bias" in state_dict:
                bias = state_dict[f"{layer}.bias"]
                size += f.write(pack("<I", bias.shape.numel()))
                size += f.write(bias.cpu().numpy().tobytes())
            else:
                size += f.write(pack("<I", 0))
    return size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "-i",
        type=str,
        default="mnist_mlp.pt",
        metavar="INPUT",
        help="path to the original pytorch model",
    )
    parser.add_argument(
        "-o",
        type=str,
        default="mnist_mlp.bin",
        metavar="OUTPUT",
        help="path to converted model",
    )

    args = parser.parse_args()
    state_dict = torch.load(args.i)
    save_model(state_dict, args.o)
