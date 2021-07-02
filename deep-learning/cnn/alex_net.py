"""
    PyTorch implementation of AlexNet (Krizhevsky et al., 2012)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Conv2dWithReLU(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return F.relu(super().forward(x))

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.00002

        # padding size for conv2d layers to make output size equal to input size is found by solving for (n + 2*pad - ks)//stride + 1 == desired_output_size
        self.layers = nn.Sequential(                  # Tensor shape afterwards (assuming 227 x 227 x 3 images)
                Conv2dWithReLU(3, 96, 11, 4),         # 96  x 55 x 55  
                nn.LocalResponseNorm(2, self.alpha),  # ''
                nn.MaxPool2d(3, 2),                   # 96  x 27 x 27 
                Conv2dWithReLU(96, 256, 5, 8),        # 256 x 27 x 27  --> padding = "SAME"
                nn.LocalResponseNorm(2, self.alpha),  # ''
                nn.MaxPool2d(3,2),                    # 256 x 13 x 13 
                Conv2dWithReLU(256, 384, 3, 1),       # 384 x 13 x 13  --> padding = "SAME"
                Conv2dWithReLU(384, 384, 3, 1),       # 384 x 13 x 13  --> padding = "SAME"
                Conv2dWithReLU(384, 256, 3, 1),       # 256 x 13 x 13  --> padding = "SAME"
                nn.MaxPool2d(3,2),                    # 256 x 6  x 6 
                nn.Linear(256*6*6, 4096),             # 4096 
                nn.Dropout(0.5),                      # ''
                nn.ReLU(),                            # ''
                nn.Linear(4096, 4096),                # 4096 
                nn.Dropout(0.5),                      # ''
                nn.ReLU(),                            # ''
                nn.Linear(4096, 1000),                # 1000
                nn.Softmax()                          # ''
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    print("Running on ", device)

    print("============ AlexNet details ============")
    model = AlexNet().to(device)
    print(model)

    # train network
