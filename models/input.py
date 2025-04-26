from torch.nn import Module, Bilinear, Linear
from torch import movedim, stack, cat, flatten

class InputModule(Module):
    def __init__(self, device):
        super().__init__()
        self.relationship = RelationshipModule(8, 8, 16, device)
        self.square = SquareModule(8, 1024, 32, 32, device)

    def forward(self, inp):
        # input shape n, 8, 8, 8
        # input shape n, x, y, 8
        out = self.relationship(inp)
        #  shape n, 8, 8, 8, 8, 16
        out_resize = flatten(out, 3)
        #  shape n, 8, 8, 1024
        ret = self.square(out_resize)
        #  shape n, 32, 32, 32
        return ret

class RelationshipModule(Module):
    def __init__(self, board_size, in_features, out_features, device):
        super().__init__()
        self.board_size = board_size
        self.diff_size = board_size * 2 - 1
        self.bilinears = [[Bilinear(in_features, in_features, out_features, device=device) for _ in range(self.diff_size)] for __ in range(self.diff_size)]

    def forward(self, inp):
        # input shape n, 8, 8, 8
        inp = movedim(inp, 0, 2)
        # shape 8, 8, n, 8
        # optimize later
        a_x_rets = []
        for a_x in range(self.board_size):
            a_y_rets = []
            for a_y in range(self.board_size):
                a = inp[a_x][a_y]
                b_x_rets = []
                for b_x in range(self.board_size):
                    b_y_rets = []
                    for b_y in range(self.board_size):
                        b = inp[b_x][b_y]
                        bilinear = self.bilinears[b_x - a_x + 7][b_y - a_y + 7]
                        b_y_rets.append(bilinear.forward(a, b))
                    b_x_rets.append(stack(b_y_rets, 1))
                a_y_rets.append(stack(b_x_rets, 1))
            a_x_rets.append(stack(a_y_rets, 1))
        output = stack(a_x_rets, 1)
        # output shape n, 8, 8, 8, 8, 16
        return output

class SquareModule(Module):
    def __init__(self, in_size, in_features, out_size, out_features, device):
        super().__init__()
        self.in_size = in_size
        self.in_features = in_features
        self.out_size = out_size
        self.out_features = out_features
        self.scale = self.out_size // self.in_size
        self.linears = [[Linear(self.in_features, self.out_features, device=device) for _ in range(self.scale)] for __ in range(self.scale)]

    def forward(self, inp):
        # input shape n, 8, 8, 1024
        rets_x = []
        for x in range(self.scale):
            rets_y = []
            for y in range(self.scale):
                rets_y.append(self.linears[x][y].forward(inp))
            rets_x.append(cat(rets_y, 2))
        # output shape n, 32, 32, 32
        return cat(rets_x, 1)