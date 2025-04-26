from torch.nn import Module, Linear, Conv2d, ReLU
from torch import cat, movedim

class HiddenModule(Module):
    def __init__(self, depth, device):
        super().__init__()
        self.depth = depth
        self.turnaround = Linear(32, 32, device=device)
        self.depth_finder = DepthModule(device)
        self.convergence = ConvergenceModule(device)
        self.activation = ReLU()

    def forward(self, inp):
        # shape n, 32, 32, 32d
        saved_closed = []
        open = inp
        # shape n, 32d, 32, 32
        for _ in range(self.depth):
            open, closed = self.depth_finder(open)
            saved_closed.insert(0, closed)
            open = self.activation(open)

        closing = self.turnaround(open)

        for closed in saved_closed:
            closing = self.convergence(closing, closed)
            closing = self.activation(closing)

        # shape n, 32, 32, 32d
        return closing
    
    def set_depth(self, depth):
        self.depth = depth

class DepthModule(Module):
    def __init__(self, device):
        super().__init__()
        self.conv3 = Conv2d(32, 32, 3, padding='same', groups=4, device=device)
        self.conv5 = Conv2d(32, 32, 5, padding='same', groups=4, device=device)
        self.conv7 = Conv2d(32, 32, 7, padding='same', groups=4, device=device)
        self.conv9 = Conv2d(32, 32, 9, padding='same', groups=4, device=device)
        self.closed_linear = Linear(128, 32, device=device)
        self.open_linear = Linear(128, 32, device=device)

    def forward(self, inp):
        # input shape n, 32, 32, 32
        output = movedim(inp, 3, 1)
        # input shape n, d, x, y
        output = cat((
            self.conv3(output),
            self.conv5(output),
            self.conv7(output),
            self.conv9(output)),
            1
        )
        output = movedim(output, 1, 3)
        # shape n, 128, 32, 32
        return self.open_linear(output), self.closed_linear(output)


class ConvergenceModule(Module):
    def __init__(self, device):
        super().__init__()
        self.conv3 = Conv2d(32, 16, 3, padding='same', groups=4, device=device)
        self.conv5 = Conv2d(32, 16, 5, padding='same', groups=4, device=device)
        self.conv7 = Conv2d(32, 16, 7, padding='same', groups=4, device=device)
        self.conv9 = Conv2d(32, 16, 9, padding='same', groups=4, device=device)
        self.conv32 = Conv2d(32, 16, 3, padding='same', groups=4, device=device)
        self.conv52 = Conv2d(32, 16, 5, padding='same', groups=4, device=device)
        self.conv72 = Conv2d(32, 16, 7, padding='same', groups=4, device=device)
        self.conv92 = Conv2d(32, 16, 9, padding='same', groups=4, device=device)
        self.closing_linear = Linear(128, 32, device=device)

    def forward(self, open, closed):
        # input shape n, 32, 32, 32
        open = movedim(open, 3, 1)
        closed = movedim(closed, 3, 1)
        # input shape n, d, x, y
        output = cat((
            self.conv3(open),
            self.conv5(open),
            self.conv7(open),
            self.conv9(open),
            self.conv32(closed),
            self.conv52(closed),
            self.conv72(closed),
            self.conv92(closed)),
            1
        )
        # shape n, 128, 32, 32
        output = movedim(output, 1, 3)
        return self.closing_linear(output)