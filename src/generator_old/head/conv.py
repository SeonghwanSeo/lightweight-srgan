from torch import nn

class ConvHead(nn.Module) :
    def __init__(self, kernel_size = 9) :
        super().__init__()
        padding = (kernel_size - 1) // 2
        head = nn.Conv2d(64, 3, kernel_size, padding=padding)
        self.head = head
    def forward(self, x) :
        return self.head(x)
