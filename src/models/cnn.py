import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleInsectCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(SimpleInsectCNN, self).__init__()
        # Input: 3 x 224 x 224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 32 x 112 x 112
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64 x 56 x 56
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 128 x 28 x 28
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # 256 x 14 x 14
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.5)
        
        # Flatten size: 256 * 14 * 14 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Layer 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x) # Output logits (no softmax here, dùng nn.CrossEntropyLoss bên ngoài đã bao gồm Softmax)
        return x

if __name__ == "__main__":
    # Test model input/output dimension
    model = SimpleInsectCNN(num_classes=13)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Mô hình hoạt động tốt! Kích thước Output:", output.shape)
