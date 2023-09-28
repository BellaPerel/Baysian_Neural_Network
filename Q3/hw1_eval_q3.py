import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator
from scipy.stats import bernoulli
from torch.autograd import Variable
from matplotlib import pyplot as plt
import pickle



class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1   = BayesianLinear(256, 120)
        self.fc2   = BayesianLinear(120, 84)
        self.fc3   = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class BayesianCNN_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1   = BayesianLinear(256, 100)
        self.fc2   = BayesianLinear(100, 10)
        #self.fc3   = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        #out = F.relu(self.fc2(out))
        #out = self.fc3(out)
        out = self.fc2(out)
        return out

class BayesianCNN_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1   = BayesianLinear(256, 10)
        #self.fc2   = BayesianLinear(120, 84)
        #self.fc3   = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        #out = F.relu(self.fc1(out))
        #out = F.relu(self.fc2(out))
        #out = self.fc3(out)
        out = self.fc1(out)
        return out

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def evaluate_hw1():

    test_dataset = dsets.MNIST(root="./data",
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True
                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_dataset.__len__(),
                                              shuffle=True)

    test_features, test_labels = next(iter(test_loader))


    with open('model_pkl.pkl' , 'rb') as f:
        model_LR = pickle.load(f)
    accuracy_test = 0
    count_test = 0
    test_features = Variable(test_features.view(-1, 28*28))
    outputs = model_LR(test_features)
    _, predicted = torch.max(outputs.data, 1)
    count_test += test_labels.size(0)

    accuracy_test += (predicted == test_labels).sum().item()
    accuracy_test = 100 * accuracy_test/count_test
    print("Test accuracy for logistic regression: ", accuracy_test)

    with open('model_pkl_2.pkl', 'rb') as f:
        model_LR_L2 = pickle.load(f)
    accuracy_test = 0
    count_test = 0
    test_features = Variable(test_features.view(-1, 28 * 28))
    outputs = model_LR_L2(test_features)
    _, predicted = torch.max(outputs.data, 1)
    count_test += test_labels.size(0)
    # for gpu, bring the predicted and labels back to cpu fro python operations to work
    accuracy_test += (predicted == test_labels).sum().item()
    accuracy_test = 100 * accuracy_test / count_test
    print("Test accuracy for logistic regression with regularizer: ", accuracy_test)

    ##bysian 3 layers

    train_dataset = dsets.MNIST(root="./data",
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True
                                )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=200,
                                               shuffle=True)

    test_dataset = dsets.MNIST(root="./data",
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True
                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_dataset.__len__(),
                                              shuffle=True)

    test_features, test_labels = next(iter(test_loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = BayesianCNN().to(device)
    model_all = torch.load("model_pkl_3.pkl")
    model_all = model_all[list(model_all.keys())[0]]
    classifier.conv1 = model_all.conv1
    classifier.conv2 = model_all.conv2
    classifier.fc1 = model_all.fc1
    classifier.fc2 = model_all.fc2
    classifier.fc3 = model_all.fc3


    iteration = 0
    count_test = 0
    accuracy_test =0
    # mnist full data



    outputs = classifier(test_features.to(device))
    _, predicted = torch.max(outputs.data, 1)
    count_test += test_labels.size(0)
    accuracy_test += (predicted == test_labels.to(device)).sum().item()

    print("Test accuracy for bysian 3 layers is" , 100 * accuracy_test / count_test)

    ##bysian 2 layers
    accuracy_test =0
    count_test = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_2 = BayesianCNN_2().to(device)

    weights2 = torch.load('model_pkl_4.pkl')
    weights2 = weights2[list(weights2.keys())[0]]
    model_2.conv1 = weights2.conv1
    model_2.conv2 = weights2.conv2
    model_2.fc1 = weights2.fc1
    model_2.fc2 = weights2.fc2
    # model_2.fc3 = weights2.fc3

    iteration = 0



    outputs = model_2(test_features.to(device))
    _, predicted = torch.max(outputs.data, 1)
    count_test += test_labels.size(0)
    accuracy_test += (predicted == test_labels.to(device)).sum().item()
    print("Test accuracy for baysian 2 layers is" , 100 * accuracy_test / count_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_3= BayesianCNN_3().to(device)

    weights3 = torch.load('model_pkl_5.pkl')
    weights3 = weights3[list(weights3.keys())[0]]
    model_3.conv1 = weights3.conv1
    model_3.conv2 = weights3.conv2
    model_3.fc1 = weights3.fc1
    # model_3.fc2 = weights3.fc2
    # model_3.fc3 = weights3.fc3

    iteration = 0
    outputs = model_3(test_features.to(device))
    _, predicted = torch.max(outputs.data, 1)
    count_test += test_labels.size(0)
    accuracy_test += (predicted == test_labels.to(device)).sum().item()
    print("Test accuracy for baysian 1 layers is" , 100 * accuracy_test / count_test)




evaluate_hw1()