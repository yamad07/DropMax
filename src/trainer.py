import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Trainer(object):

    def __init__(self, model, trainloader, testloader, n_epoch, lr):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.n_epoch = n_epoch
        self.optim = optim.SGD(self.model.parameters(), lr=lr)

    def train(self):
        for epoch in range(self.n_epoch):
            self.model.train()
            for i, (images, labels) in enumerate(self.trainloader):
                self.optim.zero_grad()
                preds = self.model(images)
                loss = F.nll_loss(preds, labels)

                loss.backward()
                self.optim.step()
            print("Epoch: {} Loss: {}".format(epoch, loss))
