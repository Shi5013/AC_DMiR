import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, stride=1):
        super(BottleneckBlock, self).__init__()

        # First 3x3x3 convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second 3x3x3 convolution
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 1x1x1 convolution for residual connection
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.residual_bn = nn.BatchNorm3d(out_channels)

        # Final 3x3x3 convolution for segmentation task
        self.reg_seg_conv = nn.Conv3d(out_channels, num_classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x

        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        residual = self.residual_conv(residual)
        residual = self.residual_bn(residual)
        out += residual
        out = self.relu(out)

        # Segmentation task final convolution
        bottleneck_output = self.reg_seg_conv(out)

        return bottleneck_output

# Example usage:
in_channels = 16  # Adjust based on your specific input channels
out_channels = 16  # Adjust based on your desired output channels
num_classes = 8  # Adjust based on your number of classes for segmentation

bottleneck = BottleneckBlock(in_channels, out_channels, num_classes)

# Example input tensor (adjust the dimensions as per your input)
input_tensor = torch.randn((1, in_channels, 32, 64, 64))

# Forward pass through the bottleneck block
output = bottleneck(input_tensor)

# Print the output shape
print("Segmentation Output Shape:", output.shape)

# Segmentation Output Shape: torch.Size([1, 8, 32, 64, 64])