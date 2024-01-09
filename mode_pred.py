# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv
import gc
import copy

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py


num_classes = 3

# Create CNN Model
class CNN3DModel_best(nn.Module):
    def __init__(self):
        super(CNN3DModel_best, self).__init__()
        
        self.conv_layer_set1 = nn.Sequential(
            nn.Conv3d(3, 128, kernel_size=(5, 5, 5), stride=2, padding=2),
            nn.LeakyReLU(),
            )
        self.conv_layer_set2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
            )
        self.conv_layer_set3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
            )
        self.fc1_mode = nn.Linear(6*2*2*512, 512)
        self.fc2_mode = nn.Linear(512, 64)
        self.fc3_mode = nn.Linear(64, num_classes)
        self.relu_mode = nn.LeakyReLU()
        self.batch1_mode = nn.BatchNorm1d(512)
        self.batch2_mode = nn.BatchNorm1d(64)
        self.drop_mode = nn.Dropout(p=0.25)        

    def forward(self, x):
        out = self.conv_layer_set1(x)
        out = self.conv_layer_set2(out)
        out = self.conv_layer_set3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1_mode(out)
        out = self.relu_mode(out)
        out = self.batch1_mode(out)
        out = self.drop_mode(out)
        out = self.fc2_mode(out)
        out = self.relu_mode(out)
        out = self.batch2_mode(out)
        out = self.drop_mode(out)
        out = self.fc3_mode(out)
        
        return out

# Create CNN
model_best = CNN3DModel_best()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()


model_savepath = 'D:/School/UofT/Research/3DCNNModel/best_model_3_layers_all_128_512_b64_lr2e4.pt'
model_best.load_state_dict(torch.load(model_savepath,map_location=torch.device('cpu')), strict=False)
model_best.eval()

valid_size = 64*15
test_size = 1
left_valid_size = 64*10


def predict(data):
    #print(data.size)
    #print(data.shape)
    data = data.reshape((192,64,64,3))

    X_train_all_torch = torch.from_numpy(data)

    test = Variable(X_train_all_torch.float().view(1,3,192,64,64))
    output = model_best(test)
    prob = F.softmax(output, dim=0)
    #select index with maximum prediction score
    pred = torch.max(output.data, 1)[1]
    return pred
