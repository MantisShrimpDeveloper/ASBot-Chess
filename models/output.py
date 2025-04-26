from torch.nn import Module, Linear, Sigmoid
from torch import split, cat, reshape, squeeze

class OutputModule(Module):
    def __init__(self, device):
        super().__init__()
        self.splitter = Splitter(8, 512, device)
        self.move_reduction = MoveReduction(device)

    def forward(self, inp):
        out = self.splitter(inp)
        out = self.move_reduction(out)
        return out

class Splitter(Module):
    def __init__(self, out_size, out_features, device):
        super().__init__()
        self.out_size = out_size
        self.out_features = out_features
        self.linear = Linear(out_features, out_features, device=device)

    def forward(self, inp):
        # input shape n, 32, 32, 32
        out = cat(split(inp, self.out_size, 2), 3)
        out = cat(split(out, self.out_size, 1), 3)
        # shape n, 8, 8, 512
        out = self.linear(out)
        # shape n, 8, 8, 512
        return out

class MoveReduction(Module):
    def __init__(self, device):
        super().__init__()
        self.final_linear = Linear(8, 1, device=device)
        self.sigmoid = Sigmoid()

    def forward(self, inp):
        # input shape n, 8, 8, 512
        out = reshape(inp, (-1, 8, 8, 8, 8, 8))
        # shape n, 8, 8, 8, 8, 8
        out = self.final_linear(out)
        # shape n, 8, 8, 8, 8, 1
        out = squeeze(out, 5)
        # shape n, 8, 8, 8, 8
        return out