from torch.nn import Module
from models.input import InputModule
from models.hidden import HiddenModule
from models.output import OutputModule

class Network(Module):
    def __init__(self, device):
        super().__init__()
        self.input_module = InputModule(device)
        self.middle_module = HiddenModule(10, device)
        self.ouput_module = OutputModule(device)

    def set_depth(self, depth):
        self.middle_module.set_depth(depth)

    def forward(self, x):
        # input shape n, 8, 8, 8
        x = self.input_module(x)
        #  shape n, 32, 32, 32
        x = self.middle_module(x)
        #  shape n, 32, 32, 32
        x = self.ouput_module(x)
        # shape n, 8, 8, 8, 8
        return x