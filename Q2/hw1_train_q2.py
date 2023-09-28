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


train_dataset = dsets.MNIST(root="./data",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="./data",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True
                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=True)

@variational_estimator
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


#plot the acuuracy of train and test as function of epochs with ticks of 1 on x axes and titles
def plot_accuracy(train_acc, test_acc, num_epochs, title):
    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.legend()
    plt.title('Accuracy of train and test '+ title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #if num of epochs is bigger than 50, then the ticks will be 5
    if num_epochs > 50:
        plt.xticks(np.arange(0, num_epochs, 5))
    else:
        plt.xticks(np.arange(0, num_epochs, 1))
    plt.show()

############
#Q 2.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianCNN().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
kl_list = []
for epoch in range(100):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    for i, (images, lables) in enumerate(train_loader):
        optimizer.zero_grad() #todo: what
        loss = classifier.sample_elbo(inputs=images.to(device),
                                      labels=lables.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1/50000)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        outputs = classifier(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        accuracy_train = (predicted == lables).sum().item() + accuracy_train
        # accuracy_train = np.float16(accuracy_train)
        # accuracy_test = np.float16(accuracy_test)
        count_train = count_train + images.size(0)
    for i, (images2, labels2) in enumerate(test_loader):
        outputs2 = classifier(images2.to(device))
        _, predicted2 = torch.max(outputs2.data, 1)
        accuracy_test = (predicted2 == labels2).sum().item() + accuracy_test
        count_test = count_test + images2.size(0)
    #save the KL divergence in list
    kl_list.append(kl_divergence_from_nn(classifier))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    #plot the acuuracy of train and test as function of epochs
plot_accuracy(train_acc, test_acc, 100, "Q2.1")
num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print(kl_divergence_from_nn(classifier))
print(kl_divergence_from_nn(classifier)/(num_params))
torch.save({"AllData": classifier}, 'AllData.pkl')
#plot kl divergence as function of epochs
"""
plt.plot(kl_list, label='KL divergence')
plt.legend()
plt.title('KL divergence')
plt.xlabel('Epochs')
plt.ylabel('KL divergence')
plt.xticks(np.arange(0, 4, 5))
plt.show()
"""

def load_data_200():
    batch_size = 200

    transform = transforms.Compose([transforms.ToTensor()])

    # Loading the data and splitting to train and test
    train_set = dsets.MNIST('/files/',train = True,transform=transform,download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size, shuffle=False)

    test_set = dsets.MNIST('/files/',train = False,transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size= test_set.__len__(),shuffle=False)
    train = enumerate(train_loader)
    _, (train_images, labels_train) = next(train)
    test = enumerate(test_loader)
    _, (test_images, labels_test) = next(test)
    return train_images, labels_train, test_images, labels_test, train_loader, test_loader

#load data 200 samples for train but only take the samples with lables 8 and 3
def load_data_200_83():
    #load data
    train_images, labels_train, test_images, labels_test, train_loader, test_loader = load_data_200()
    #take only the samples with lables 8 and 3
    train_images_83 = train_images[(labels_train == 8) | (labels_train == 3)]
    labels_train_83 = labels_train[(labels_train == 8) | (labels_train == 3)]
    test_images_83 = test_images[(labels_test == 8) | (labels_test == 3)]
    labels_test_83 = labels_test[(labels_test == 8) | (labels_test == 3)]
    return train_images_83, labels_train_83, test_images_83, labels_test_83, train_loader, test_loader


def load_data_200_random():
    batch_size = 200

    transform = transforms.Compose([transforms.ToTensor()])

    # Loading the data and splitting to train and test
    train_set = dsets.MNIST('/files/',train = True,transform=transform,download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size, shuffle=False)
    train = enumerate(train_loader)
    _, (train_images, labels_train) = next(train)
    random_labels_train = bernoulli.rvs(0.5, size=200)
    random_labels_train_tensor = torch.Tensor(random_labels_train).long()

    test_set = dsets.MNIST('/files/',train = False,transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size= test_set.__len__(),shuffle=False)
    test = enumerate(test_loader)
    _, (test_images, labels_test) = next(test)
    random_labels_test = bernoulli.rvs(0.5, size=len(test_images))
    random_labels_test_tensor = torch.Tensor(random_labels_test).long()

    return train_images, random_labels_train_tensor, test_images, random_labels_test_tensor, train_loader, test_loader

###################################################
#train on 200 samples
###################################################
#train_loader, test_loader = load_data_200()
print("train on 200 samples - not random!!!")
train_images, labels_train_tensor, test_images, labels_test_tensor, train_loader,\
    test_loader = load_data_200()

classifier_2 = BayesianCNN().to(device)
optimizer = optim.Adam(classifier_2.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
for epoch in range(100):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    #for i, (images, lables) in enumerate(train_loader):
    optimizer.zero_grad() #todo: what
    loss = classifier_2.sample_elbo(inputs=train_images.to(device),
                                      labels=labels_train_tensor.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1/50000)
        #print(loss)
    loss.backward()
    optimizer.step() #todo: what is it
    outputs = classifier_2(train_images.to(device))
    outputs2 = classifier_2(test_images.to(device))
    _, predicted = torch.max(outputs.data, 1)
    _, predicted2 = torch.max(outputs2.data, 1)

    accuracy_train = (predicted == labels_train_tensor).sum()+accuracy_train
    accuracy_train = np.float16(accuracy_train)
    accuracy_test = (predicted2 == labels_test_tensor).sum()+accuracy_test
    accuracy_test = np.float32(accuracy_test)
    count_train = count_train + train_images.size(0)
    count_test = count_test + test_images.size(0)
    print(kl_divergence_from_nn(classifier_2))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)
    #plot the acuuracy of train and test
num_params = sum(p.numel() for p in classifier_2.parameters() if p.requires_grad)
print(kl_divergence_from_nn(classifier_2))
print(kl_divergence_from_nn(classifier_2)/(num_params))
torch.save({"First200": classifier_2}, 'First200.pkl')
"""
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()
"""
plot_accuracy(train_acc, test_acc, 100, "train on 200 samples - Q2.2")

#####################################
#train on samples with lables 8 and 3
###################################
#train_loader, test_loader = load_data_200_83()
train_images, labels_train_tensor, test_images, labels_test_tensor, train_loader,\
    test_loader = load_data_200_83()
#write the same baysian classifier names classifier_5 as below but with 2 output neurons
classifier_5 = BayesianCNN().to(device)
optimizer = optim.Adam(classifier_5.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
for epoch in range(100):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    #for i, (images, lables) in enumerate(train_loader):
    optimizer.zero_grad() #todo: what
    loss = classifier_5.sample_elbo(inputs=train_images.to(device),
                                      labels=labels_train_tensor.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1/50000)
        #print(loss)
    loss.backward()
    optimizer.step() #todo: what is it
    outputs = classifier_5(train_images.to(device))
    outputs2 = classifier_5(test_images.to(device))
    _, predicted = torch.max(outputs.data, 1)
    _, predicted2 = torch.max(outputs2.data, 1)

    accuracy_train = (predicted == labels_train_tensor).sum()+accuracy_train
    accuracy_train = np.float16(accuracy_train)
    accuracy_test = (predicted2 == labels_test_tensor).sum()+accuracy_test
    accuracy_test = np.float32(accuracy_test)
    count_train = count_train + train_images.size(0)
    count_test = count_test + test_images.size(0)
    print(kl_divergence_from_nn(classifier_5))
    #print(accuracy_train/count_train)
    #print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)
#plot the acuuracy of train and test
#add titles to plot
num_params = sum(p.numel() for p in classifier_5.parameters() if p.requires_grad)
print(kl_divergence_from_nn(classifier_5))
print(kl_divergence_from_nn(classifier_5)/(num_params))
torch.save({"First20083": classifier_5}, 'First20083.pkl')
plot_accuracy(train_acc, test_acc, 100, "train on 200 samples with lables 8 and 3 - Q2.3")
"""
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()
"""
#load all train data but only with lables 8 and 3
train_images, labels_train_tensor, test_images, labels_test_tensor, train_loader,\
    test_loader = load_data_200_83()
#load all test data but only with lables 8 and 3
train_images, labels_train_tensor, test_images, labels_test_tensor, train_loader,\
    test_loader = load_data_200_83()
#load ALL test data but only with lables 8 and 3
train_images, labels_train_tensor, test_images, labels_test_tensor, train_loader,\
    test_loader = load_data_200_83()

#####################################
train_dataset = dsets.MNIST(root="./data",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True
                            )
filter_labels = [8, 3]
# filtering the training data
filter_indices = np.where((train_dataset.targets == filter_labels[0]) | (train_dataset.targets == filter_labels[1]))
# filter_indices = np.where(train_dataset.targets in filter)
train_dataset.data = train_dataset.data[filter_indices[0], :, :]
train_dataset.targets = train_dataset.targets[filter_indices]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)



test_dataset = dsets.MNIST(root="./data",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True
                           )

# filter the test data as well
# filtering the training data
filter_indices = np.where((test_dataset.targets == filter_labels[0]) | (test_dataset.targets == filter_labels[1]))
# filter_indices = np.where(train_dataset.targets in filter)
test_dataset.data = test_dataset.data[filter_indices[0], :, :]
test_dataset.targets = test_dataset.targets[filter_indices]

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=False)


#build classifier_7 with 2 output neurons and train on all of the above sampeles usong the loop of classifier_
classifier_7 = BayesianCNN().to(device)
optimizer = optim.Adam(classifier_7.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0

for epoch in range(4):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0
    for i, (images, lables) in enumerate(train_loader):
        optimizer.zero_grad() #todo: what
        loss = classifier_7.sample_elbo(inputs=images.to(device),
                                      labels=lables.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1/50000)
        #print(loss)
        loss.backward()
        optimizer.step() #todo: what is it
        outputs = classifier_7(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        accuracy_train = (predicted == lables).sum().item() + accuracy_train
        # accuracy_train = np.float16(accuracy_train)
        # accuracy_test = np.float16(accuracy_test)
        count_train = count_train + images.size(0)
    for i, (images2, labels2) in enumerate(test_loader):
        outputs2 = classifier_7(images2.to(device))
        _, predicted2 = torch.max(outputs2.data, 1)
        accuracy_test = (predicted2 == labels2).sum().item() + accuracy_test
        count_test = count_test + images2.size(0)
    print(kl_divergence_from_nn(classifier_7))
    print(accuracy_train/count_train)
    print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)
    #what is the hidden size of the model?
    #print(classifier.fc1.hidden_size)
    #plot the acuuracy of train and test as function of epochs
num_params = sum(p.numel() for p in classifier_2.parameters() if p.requires_grad)
print(kl_divergence_from_nn(classifier_2))
print(kl_divergence_from_nn(classifier_2)/(num_params))
torch.save({"all83": classifier_2}, 'all83.pkl')
plot_accuracy(train_acc, test_acc, 4, "train on all samples with lables 8 and 3 - Q2.4")


#####################################3
#random 200 samples
##############################################
print("random")
train_images, labels_train_tensor, test_images, labels_test_tensor, train_loader, \
test_loader = load_data_200_random()
classifier_3 = BayesianCNN().to(device)
optimizer = optim.Adam(classifier_3.parameters(), lr=0.001) #todo: what
criterion = torch.nn.CrossEntropyLoss() # todo: what is this?
train_acc = []
test_acc = []
iteration = 0
for epoch in range(100):
    count_train = 0
    count_test = 0
    accuracy_test = 0
    accuracy_train = 0

    optimizer.zero_grad() #todo: what
    loss = classifier_3.sample_elbo(inputs=train_images.to(device),
                                        labels=labels_train_tensor.to(device),
                                        criterion=criterion,
                                        sample_nbr=3,
                                        complexity_cost_weight=1/50000)
    #print(loss)
    loss.backward()
    optimizer.step() #todo: what is it
    outputs = classifier_3(train_images.to(device))
    outputs2 = classifier_3(test_images.to(device))
    _, predicted = torch.max(outputs.data, 1)
    _, predicted2 = torch.max(outputs2.data, 1)

    accuracy_train = (predicted == labels_train_tensor).sum()+accuracy_train
    accuracy_train = np.float16(accuracy_train)
    accuracy_test = (predicted2 == labels_test_tensor).sum()+accuracy_test
    accuracy_test = np.float32(accuracy_test)
    count_train = count_train + train_images.size(0)
    count_test = count_test + test_images.size(0)
    print(kl_divergence_from_nn(classifier_3))
    print(accuracy_train/count_train)
    print(accuracy_test/count_test)
    train_acc.append(accuracy_train/count_train)
    test_acc.append(accuracy_test/count_test)

num_params = sum(p.numel() for p in classifier_3.parameters() if p.requires_grad)
print(kl_divergence_from_nn(classifier_3))
print(kl_divergence_from_nn(classifier_3)/(num_params))
torch.save({"random200": classifier_3}, 'random200.pkl')
plot_accuracy(train_acc, test_acc, 100, "random 200 samples - Q2.5")
"""
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()
"""
