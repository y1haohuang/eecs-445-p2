'''
EECS 445 - Introduction to Machine Learning
Winter 2020 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        #

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)


        for fc in [self.fc1, self.fc2, self.fc3]:
            print(fc.weight)
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)



    def forward(self, x):
        N, C, H, W = x.shape

        # TODO: forward pass
        conv1_res = F.relu(self.conv1(x))
        conv2_res = F.relu(self.conv2(conv1_res))
        conv3_res = F.relu(self.conv3(conv2_res))

        conv3_res = conv3_res.view(-1, 512)

        x = F.relu(self.fc1(conv3_res))
        y = F.relu(self.fc2(x))
        z = self.fc3(y)
        #

        return z


class Challenge_deeper(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        self.dropout1 = torch.nn.Dropout(p=0.3, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=0.3, inplace=False)
        #

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)


        for fc in [self.fc1, self.fc2, self.fc3]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)



    def forward(self, x):
        N, C, H, W = x.shape

        # TODO: forward pass
        conv1_res = F.relu(self.conv1(x))
        conv2_res = F.relu(self.conv2(conv1_res))
        conv3_res = F.relu(self.conv3(conv2_res))
        conv4_res = F.relu(self.conv4(conv3_res))
        conv5_res = F.relu(self.conv5(conv4_res))

        conv5_res = conv5_res.view(-1, 512)

        conv5_res = self.dropout1(conv5_res)
        x = F.relu(self.fc1(conv5_res))
        x = self.dropout2(x)
        y = F.relu(self.fc2(x))
        z = self.fc3(y)
        #

        return z




class Challenge_try(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv5 =  nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        self.dropout1 = torch.nn.Dropout(p=0.5, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=0.5, inplace=False)
        #

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc1, self.fc2, self.fc3]:
            print(fc.weight)
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape

        # TODO: forward pass
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = self.maxpool1(z)
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = self.maxpool2(z)
        z = F.relu(self.conv5(z))
        z = F.relu(self.conv6(z))
        z = self.maxpool3(z)
        z = self.adaptiveavgpool(z)

        z = z.view(-1, 1024)

        z = self.dropout1(z)
        z = F.relu(self.fc1(z))
        z = self.dropout2(z)
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        #

        return z