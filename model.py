import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Block 4
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Block 5
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        # Dropout layers
        self.dropout_fc = nn.Dropout(0.5)   # Dropout for fully connected layers
        
        # Fully connected layer
        self.fc1 = nn.Linear(512 * 4 * 4 , 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pre_process(x)
        # Block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 128 -> 64

        
        # Block 2
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 64 -> 32

        
        # Block 3
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 32 -> 16


        # Block 4
        x = F.leaky_relu(self.bn7(self.conv7(x)))
        x = F.leaky_relu(self.bn8(self.conv8(x)))  # 16 -> 8
        x = F.max_pool2d(x, kernel_size=2, stride=2)  

        # Block 5
        x = F.leaky_relu(self.bn9(self.conv9(x)))
        x = F.leaky_relu(self.bn10(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 8 -> 4
        
        # Flatten
        x = x.view(x.size(0), -1)  # batch_size x (512 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)  # Apply dropout
        x = self.fc2(x)
        return x

    def pre_process(self, x):
        return x.float()