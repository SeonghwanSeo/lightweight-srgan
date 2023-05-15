from torch import nn

class DoubleConvHead(nn.Module) :
    def __init__(self, kernel_size = 9) :
        super().__init__()
        padding = (kernel_size - 1) // 2
        head = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size, padding=padding),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size, padding=padding),
        )
        self.head = head
    def forward(self, x) :
        return self.head(x)
