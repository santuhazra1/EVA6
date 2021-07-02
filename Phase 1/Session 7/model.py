import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block (C1)
        self.convblock1 = nn.Sequential(
            # Normal Conv
            nn.Conv2d(3, 30, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(30),
        
            # Depthwise Saperable
            nn.Conv2d(30, 30, 3, padding=1, groups=30, bias=False),
            nn.Conv2d(30, 60, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(60),
            
        )

        # Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(60, 30, 1, padding=0, bias=False),
        )

        # Convolution Block 2 (C2)
        self.convblock2 = nn.Sequential(
            # Normal Conv
            nn.Conv2d(30, 30, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            
            # Depthwise saperable
            nn.Conv2d(30, 30, 3, padding=1, groups=30, bias=False),
            nn.Conv2d(30, 60, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(60),

            # Dialation Conv
            nn.Conv2d(60, 60, 3, dilation=4, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(60),

        )

        # Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(60, 30, 1, padding=0, bias=False),
        )

        #  Convolution Block 3 (C3)
        # Depthwise Separable Convolution
        self.convblock3 = nn.Sequential(
            nn.Conv2d(30, 30, 3, padding=1, groups=30, bias=False),
            nn.Conv2d(30, 60, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(60),

            nn.Conv2d(60, 60, 3, padding=1, groups=60, bias=False),
            nn.Conv2d(60, 80, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(80),
        )

        # Transition Block 3
        self.transblock3 = nn.Sequential(
            nn.Conv2d(80, 20, 1, padding=0, bias=False),
        )

        # Convolution Block 4 (C4)
        self.convblock4 = nn.Sequential(
            # Normal Conv
            nn.Conv2d(20, 20, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            
            # Depthwise saperable
            nn.Conv2d(20, 20, 3, padding=1, groups=20, bias=False),
            nn.Conv2d(20, 40, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(40),

            # Dialation Conv
            nn.Conv2d(40, 40, 3, dilation=4, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(40),

        )

        # Output Block 
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(40, 10, 1, padding=0, bias=False) 
        )

    def forward(self, x):
        
        # Input Block: Convolution Block 1
        x = self.convblock1(x)
        
        # Transition Block 1
        x = self.transblock1(x)
        
        # Convolution Block 2
        x = self.convblock2(x)
        
        # Transition Block 2
        x = self.transblock2(x)
        
        # Convolution Block 3
        x = self.convblock3(x)

        # Transition Block 3
        x = self.transblock3(x)
        
        # Convolution Block 4
        x = self.convblock4(x)  

        # Output Block
        x = self.out(x) 

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)