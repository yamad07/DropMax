import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from src.trainer import Trainer
from src.model import CNN

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=96,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=2)

trainer = Trainer(
        model=CNN(),
        trainloader=trainloader,
        testloader=testloader,
        n_epoch=5,
        lr=0.001,
        )

trainer.train()
