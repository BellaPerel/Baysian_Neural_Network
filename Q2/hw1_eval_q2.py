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


class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1   = BayesianLinear(256, 120)
        self.fc2   = BayesianLinear(120, 84)
        self.fc3   = BayesianLinear(84, 10)
    #
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

def evaluate_hw1():
    class BayesianCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = BayesianConv2d(1, 6, (5,5))
            self.conv2 = BayesianConv2d(6, 16, (5,5))
            self.fc1   = BayesianLinear(256, 120)
            self.fc2   = BayesianLinear(120, 84)
            self.fc3   = BayesianLinear(84, 10)
    #
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

    if __name__ == '__main__':
    #all_data
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
        model_all = torch.load('AllData.pkl')
        model_all = model_all[list(model_all.keys())[0]]
        classifier.conv1 = model_all.conv1
        classifier.conv2 = model_all.conv2
        classifier.fc1 = model_all.fc1
        classifier.fc2 = model_all.fc2
        classifier.fc3 = model_all.fc3

        outputs = classifier(test_features.to(device))
        _, predicted = torch.max(outputs.data, 1)

        kl_all = kl_divergence_from_nn(classifier)
        parmetrs_all  = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        avg_kl_all = kl_all/parmetrs_all
        print("kl divergence from MNIST data set is: ", kl_divergence_from_nn(classifier))
        print("number of parmetrs: ", parmetrs_all)
        print("kl avg all: ", avg_kl_all)


        #200 not random
        test_dataset = dsets.MNIST(root="./data",
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True
                                   )
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=test_dataset.__len__(),
                                                  shuffle=True)

        test = enumerate(test_loader)
        _, (test_images, labels_test_200) = next(test)



        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        classifier_200 = BayesianCNN().to(device)
        model_200 = torch.load('First200.pkl')
        model_200 = model_200[list(model_200.keys())[0]]
        classifier_200.conv1 = model_200.conv1
        classifier_200.conv2 = model_200.conv2
        classifier_200.fc1 = model_200.fc1
        classifier_200.fc2 = model_200.fc2
        classifier_200.fc3 = model_200.fc3



        # with torch.no_grad():
        outputs = classifier_200(test_images.to(device))
        _, predicted = torch.max(outputs.data, 1)


        # torch.save({"first200": classifier_200}, 'first200.pkl')

        kl_first_200 = kl_divergence_from_nn(classifier_200)
        parmetrs_200  = sum(p.numel() for p in classifier_200.parameters() if p.requires_grad)
        avg_kl_first_200= kl_first_200/parmetrs_200
        print("kl divergence from MNIST data after training only on 200 data-point is: ", kl_divergence_from_nn(classifier_200))
        print("number of parmetrs: ", parmetrs_200)
        print("kl avg all: ", avg_kl_first_200)
        #200 of 3 and 8
        test_dataset = dsets.MNIST(root="./data",
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True
                                   )

        idx = (test_dataset.targets == 8) | (test_dataset.targets == 3)
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=200,
                                                  shuffle=True)



        test = enumerate(test_loader)
        _, (test_images, labels_test_200) = next(test)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier_200_3_8 = BayesianCNN().to(device)
        model_20038 = torch.load("First20083.pkl")
        model_20038 = model_20038[list(model_20038.keys())[0]]
        classifier_200_3_8.conv1 = model_20038.conv1
        classifier_200_3_8.conv2 = model_20038.conv2
        classifier_200_3_8.fc1 = model_20038.fc1
        classifier_200_3_8.fc2 = model_20038.fc2
        classifier_200_3_8.fc3 = model_20038.fc3


        # with torch.no_grad():
        outputs = classifier_200_3_8(test_images.to(device))
        _, predicted = torch.max(outputs.data, 1)

        print("kl divergence from MNIST data after training only on 200 data-point is: ", kl_divergence_from_nn(classifier_200_3_8))

        #  torch.save({"first20038": classifier_200_3_8}, 'first20038.pkl')

        kl_200_3_8 = kl_divergence_from_nn(classifier_200_3_8)
        parmetrs_200_rand = sum(p.numel() for p in classifier_200_3_8.parameters() if p.requires_grad)
        avg_kl_200_3_8 = kl_200_3_8 / parmetrs_200_rand

        print("number of parmetrs: ", parmetrs_200_rand)
        print("kl avg all: ", avg_kl_200_3_8)

        #all data of 3 and 8
        test_dataset = dsets.MNIST(root="./data",
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True
                                   )

        idx = (test_dataset.targets==8) | (test_dataset.targets==3)
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=test_dataset.__len__(),
                                                  shuffle=True)



        test = enumerate(test_loader)
        _, (test_images, labels_test_200) = next(test)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier_3_8 = BayesianCNN().to(device)
        model_38 = torch.load("all83.pkl")
        model_38 = model_38[list(model_38.keys())[0]]
        classifier_200_3_8.conv1 = model_38.conv1
        classifier_200_3_8.conv2 = model_38.conv2
        classifier_200_3_8.fc1 = model_38.fc1
        classifier_200_3_8.fc2 = model_38.fc2
        classifier_200_3_8.fc3 = model_38.fc3


        test_features, test_labels = next(iter(test_loader))




        outputs = classifier_3_8(test_features.to(device))
        _, predicted = torch.max(outputs.data, 1)

        print("kl divergence from MNIST data set is: ", kl_divergence_from_nn(classifier_3_8))




        #torch.save({"classifier_3_8": classifier_3_8}, 'classifier_3_8.pkl')


        kl_al_38= kl_divergence_from_nn(classifier_3_8)
        parmetrs_38  = sum(p.numel() for p in classifier_3_8.parameters() if p.requires_grad)
        avg_kl_200rand = kl_al_38/kl_al_38

        print("number of parmetrs: ", parmetrs_38)
        print("kl avg all: ", avg_kl_200rand)

        #200 random
        test_dataset = dsets.MNIST(root="./data",
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True
                                       )
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=test_dataset.__len__(),
                                                      shuffle=True)
        test = enumerate(test_loader)
        _, (test_images, labels_test) = next(test)
        random_labels_test = bernoulli.rvs(0.5, size=len(test_images))
        random_labels_test = torch.Tensor(random_labels_test).long()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier_200rand = BayesianCNN().to(device)
        model_200rand = torch.load("random200.pkl")
        model_200rand = model_200rand[list(model_200rand.keys())[0]]
        classifier_200rand.conv1 = model_200rand.conv1
        classifier_200rand.conv2 = model_200rand.conv2
        classifier_200rand.fc1 = model_200rand.fc1
        classifier_200rand.fc2 = model_200rand.fc2
        classifier_200rand.fc3 = model_200rand.fc3



        outputs = classifier_200rand(test_images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        print("kl divergence from nn is: ", kl_divergence_from_nn(classifier_200rand))


        #torch.save({"first200rand": classifier_200rand}, 'first200rand.pkl')


        kl_200_random = kl_divergence_from_nn(classifier_200rand)
        parmetrs_200_rand  = sum(p.numel() for p in classifier_200rand.parameters() if p.requires_grad)
        avg_kl_200rand = kl_200_random/parmetrs_200_rand
        print("number of parmetrs: ", parmetrs_200_rand)
        print("kl 200 rand avg all: ", avg_kl_200rand)

evaluate_hw1()