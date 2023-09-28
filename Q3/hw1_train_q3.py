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
##################################################
#logistic_regression
print("linear regression")
##################################################
train_dataset = dsets.MNIST(root="/files/",
                            train=True,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="/files/",
                           train=False,
                           transform=transforms.Compose([transforms.ToTensor()]),
                           download=True
                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=True)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier_linear = LogisticRegression(28*28, 10).to(device)
optimizer = optim.Adam(classifier_linear.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
loss_func = nn.CrossEntropyLoss()

for epoch in range(100):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    # for i, ((images, lables),(images2, lables2)) in enumerate(zip(train_loader, test_loader)):
    for i, (images, lables) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        optimizer.zero_grad() #todo: what
        outputs = classifier_linear.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, lables)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        count_train = count_train + images.size(0)
        accuracy_train = (predictions == lables).sum()+accuracy_train
        accuracy_train = np.float16(accuracy_train)
    for i, (images2, labels2) in enumerate(test_loader):
        images2 = images2.view(-1, 28 * 28)
        outputs2 = classifier_linear.forward(images2)
        _, predictions2 = torch.max(outputs2.data, 1)
        accuracy_test = (predictions2 == labels2).sum()+accuracy_test
        accuracy_test = np.float32(accuracy_test)
        count_test = count_test + images2.size(0)
        # outputs = classifier_linear(images.to(device))
        # outputs2 = classifier_linear(images2.to(device))
        # _, predicted = torch.max(outputs.data, 1)
        # _, predicted2 = torch.max(outputs2.data, 1)

    #     accuracy_train = (predicted == lables).sum().item() + accuracy_train
    #     # accuracy_train = np.float16(accuracy_train)
    #     accuracy_test = (predicted2 == lables2).sum().item() + accuracy_test
    #     # accuracy_test = np.float16(accuracy_test)
    #     count_train = count_train + images.size(0)
    #     count_test = count_test + images2.size(0)
    #save accuracys in list
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)

    #print(kl_divergence_from_nn(classifier_linear))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
    #train_acc.append(accuracy_train/count_train)
    #test_acc.append(accuracy_test/count_test)
#plot accuracy of train and test vs epoch
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()
with open('model_pkl.pkl', 'wb') as files:
    pickle.dump(classifier_linear, files)


#################################################################
#logistic regression with regulariztion
##############################################################
##################################################
print("logistic regression with regulariztion")
train_dataset = dsets.MNIST(root="/files/",
                            train=True,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="/files/",
                           train=False,
                           transform=transforms.Compose([transforms.ToTensor()]),
                           download=True
                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=True)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        return torch.sigmoid(self.linear(x))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier_linear = LogisticRegression(28*28, 10).to(device)
optimizer = optim.Adam(classifier_linear.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc_2 = []
test_acc_2 = []
iteration = 0
loss_func = nn.CrossEntropyLoss()

for epoch in range(100):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    # for i, ((images, lables),(images2, lables2)) in enumerate(zip(train_loader, test_loader)):
    for i, (images, lables) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        optimizer.zero_grad() #todo: what
        outputs = classifier_linear.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        #add regularization to loss
        loss = criterion(outputs, lables) + 0.01*torch.norm(classifier_linear.linear.weight)
        #loss = criterion(outputs, lables)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        count_train = count_train + images.size(0)
        accuracy_train = (predictions == lables).sum()+accuracy_train
        accuracy_train = np.float16(accuracy_train)
    for i, (images2, labels2) in enumerate(test_loader):
        images2 = images2.view(-1, 28 * 28)
        outputs2 = classifier_linear.forward(images2)
        _, predictions2 = torch.max(outputs2.data, 1)
        accuracy_test = (predictions2 == labels2).sum()+accuracy_test
        accuracy_test = np.float32(accuracy_test)
        count_test = count_test + images2.size(0)
        # outputs = classifier_linear(images.to(device))
        # outputs2 = classifier_linear(images2.to(device))
        # _, predicted = torch.max(outputs.data, 1)
        # _, predicted2 = torch.max(outputs2.data, 1)

    #     accuracy_train = (predicted == lables).sum().item() + accuracy_train
    #     # accuracy_train = np.float16(accuracy_train)
    #     accuracy_test = (predicted2 == lables2).sum().item() + accuracy_test
    #     # accuracy_test = np.float16(accuracy_test)
    #     count_train = count_train + images.size(0)
    #     count_test = count_test + images2.size(0)
    train_acc_2.append(accuracy_train/count_train)
    test_acc_2.append(accuracy_test/count_test)
    #print(kl_divergence_from_nn(classifier_linear))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
#Plot accuracy of test among logistic regression and logistic regression with regularization vs epochs using test_acc lists
plt.plot(train_acc_2, label='train')
plt.plot(test_acc_2, label='test')
plt.legend()
plt.show()

with open('model_pkl_2.pkl', 'wb') as files:
    pickle.dump(classifier_linear, files)
#################################################################
#baysian linear
##############################################################
print("baysian linear")
train_dataset = dsets.MNIST(root="/files/",
                            train=True,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="/files/",
                           train=False,
                           transform=transforms.Compose([transforms.ToTensor()]),
                           download=True
                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=True)


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


def compute_accuracy(model, data_loader):
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianCNN().to(device)
optimizer = optim.Adam(classifier_linear.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
loss_func = nn.CrossEntropyLoss()

for epoch in range(4):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    # for i, ((images, lables),(images2, lables2)) in enumerate(zip(train_loader, test_loader)):
    for i, (images, lables) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        optimizer.zero_grad() #todo: what
        outputs = classifier_linear.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, lables)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        count_train = count_train + images.size(0)
        accuracy_train = (predictions == lables).sum()+accuracy_train
        accuracy_train = np.float16(accuracy_train)
    for i, (images2, labels2) in enumerate(test_loader):
        images2 = images2.view(-1, 28 * 28)
        outputs2 = classifier_linear.forward(images2)
        _, predictions2 = torch.max(outputs2.data, 1)
        accuracy_test = (predictions2 == labels2).sum()+accuracy_test
        accuracy_test = np.float32(accuracy_test)
        count_test = count_test + images2.size(0)
        # outputs = classifier_linear(images.to(device))
        # outputs2 = classifier_linear(images2.to(device))
        # _, predicted = torch.max(outputs.data, 1)
        # _, predicted2 = torch.max(outputs2.data, 1)

    #     accuracy_train = (predicted == lables).sum().item() + accuracy_train
    #     # accuracy_train = np.float16(accuracy_train)
    #     accuracy_test = (predicted2 == lables2).sum().item() + accuracy_test
    #     # accuracy_test = np.float16(accuracy_test)
    #     count_train = count_train + images.size(0)
    #     count_test = count_test + images2.size(0)
    #save accuracys in list
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)

    #print(kl_divergence_from_nn(classifier_linear))
    print(accuracy_train/count_train)
    print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)

print("baysian linear")
train_dataset = dsets.MNIST(root="/files/",
                            train=True,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=200,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="/files/",
                           train=False,
                           transform=transforms.Compose([transforms.ToTensor()]),
                           download=True
                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_dataset.__len__(),
                                          shuffle=True)


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


def compute_accuracy(model, data_loader):
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianCNN().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    # for i, ((images, lables),(images2, lables2)) in enumerate(zip(train_loader, test_loader)):
    for i, (images, lables) in enumerate(train_loader):
        #images = images.view(-1, 28 * 28)
        optimizer.zero_grad() #todo: what
        outputs = classifier.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, lables)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        count_train = count_train + images.size(0)
        accuracy_train = (predictions == lables).sum()+accuracy_train
        accuracy_train = np.float16(accuracy_train)
    for i, (images2, labels2) in enumerate(test_loader):
        #images2 = images2.view(-1, 28 * 28)
        outputs2 = classifier.forward(images2)
        _, predictions2 = torch.max(outputs2.data, 1)
        accuracy_test = (predictions2 == labels2).sum()+accuracy_test
        accuracy_test = np.float32(accuracy_test)
        count_test = count_test + images2.size(0)
        # outputs = classifier(images.to(device))
        # outputs2 = classifier(images2.to(device))
        # _, predicted = torch.max(outputs.data, 1)
        # _, predicted2 = torch.max(outputs2.data, 1)

    #     accuracy_train = (predicted == lables).sum().item() + accuracy_train
    #     # accuracy_train = np.float16(accuracy_train)
    #     accuracy_test = (predicted2 == lables2).sum().item() + accuracy_test
    #     # accuracy_test = np.float16(accuracy_test)
    #     count_train = count_train + images.size(0)
    #     count_test = count_test + images2.size(0)
    #save accuracys in list
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)

    #print(kl_divergence_from_nn(classifier))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()
"""
with open('model_pkl_3.pkl', 'wb') as files:
    pickle.dump(classifier, files)
"""
torch.save({"model_pkl_3": classifier}, 'model_pkl_3.pkl')

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


def compute_accuracy(model, data_loader):
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianCNN_2().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    # for i, ((images, lables),(images2, lables2)) in enumerate(zip(train_loader, test_loader)):
    for i, (images, lables) in enumerate(train_loader):
        #images = images.view(-1, 28 * 28)
        optimizer.zero_grad() #todo: what
        outputs = classifier.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, lables)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        count_train = count_train + images.size(0)
        accuracy_train = (predictions == lables).sum()+accuracy_train
        accuracy_train = np.float16(accuracy_train)
    for i, (images2, labels2) in enumerate(test_loader):
        #images2 = images2.view(-1, 28 * 28)
        outputs2 = classifier.forward(images2)
        _, predictions2 = torch.max(outputs2.data, 1)
        accuracy_test = (predictions2 == labels2).sum()+accuracy_test
        accuracy_test = np.float32(accuracy_test)
        count_test = count_test + images2.size(0)
        # outputs = classifier(images.to(device))
        # outputs2 = classifier(images2.to(device))
        # _, predicted = torch.max(outputs.data, 1)
        # _, predicted2 = torch.max(outputs2.data, 1)

    #     accuracy_train = (predicted == lables).sum().item() + accuracy_train
    #     # accuracy_train = np.float16(accuracy_train)
    #     accuracy_test = (predicted2 == lables2).sum().item() + accuracy_test
    #     # accuracy_test = np.float16(accuracy_test)
    #     count_train = count_train + images.size(0)
    #     count_test = count_test + images2.size(0)
    #save accuracys in list
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)

    #print(kl_divergence_from_nn(classifier))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()
"""
with open('model_pkl_4.pkl', 'wb') as files:
    pickle.dump(classifier, files)
"""
torch.save({"model_pkl_4": classifier}, 'model_pkl_4.pkl')

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


def compute_accuracy(model, data_loader):
    correct = 0
    total = 0
    for images, labels in data_loader:
        #images = images.view(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianCNN_3().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    # for i, ((images, lables),(images2, lables2)) in enumerate(zip(train_loader, test_loader)):
    for i, (images, lables) in enumerate(train_loader):
        #images = images.view(-1, 28 * 28)
        optimizer.zero_grad() #todo: what
        outputs = classifier.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, lables)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        count_train = count_train + images.size(0)
        accuracy_train = (predictions == lables).sum()+accuracy_train
        accuracy_train = np.float16(accuracy_train)
    for i, (images2, labels2) in enumerate(test_loader):
        #images2 = images2.view(-1, 28 * 28)
        outputs2 = classifier.forward(images2)
        _, predictions2 = torch.max(outputs2.data, 1)
        accuracy_test = (predictions2 == labels2).sum()+accuracy_test
        accuracy_test = np.float32(accuracy_test)
        count_test = count_test + images2.size(0)
        # outputs = classifier(images.to(device))
        # outputs2 = classifier(images2.to(device))
        # _, predicted = torch.max(outputs.data, 1)
        # _, predicted2 = torch.max(outputs2.data, 1)

    #     accuracy_train = (predicted == lables).sum().item() + accuracy_train
    #     # accuracy_train = np.float16(accuracy_train)
    #     accuracy_test = (predicted2 == lables2).sum().item() + accuracy_test
    #     # accuracy_test = np.float16(accuracy_test)
    #     count_train = count_train + images.size(0)
    #     count_test = count_test + images2.size(0)
    #save accuracys in list
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)

    #print(kl_divergence_from_nn(classifier))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()

torch.save({"model_pkl_5": classifier}, 'model_pkl_5.pkl')
"""
with open('model_pkl_5.pkl', 'wb') as files:
    pickle.dump(classifier, files)
"""