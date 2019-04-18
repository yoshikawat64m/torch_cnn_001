
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_finetune import make_model
from keras.datasets import fashion_mnist

import datetime
import matplotlib.pyplot as plt
import os

from MyDataset import MyDataset

from inception import inception_v3

def train(epoch):
    total_loss = 0
    total_size = 0
    #import pdb; pdb.set_trace()

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        total_size += data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            now = datetime.datetime.now()
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                now,
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), total_loss / total_size))

config = {
    'dataset_dir': 'dataset/flower_images/',
    'label_file': 'dataset/flower_images/flower_labels.csv',
    'num_classes': 10,
    'batch_size': 70,
    'num_epochs':30,
    'pretrained':True
}

train_set = MyDataset(config['label_file'], config['dataset_dir'])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = make_model('inception_v3', num_classes=10, pretrained=True, input_size=(224, 224))

model = inception_v3(pretrained=config['pretrained'], 
                     num_classes=config['num_classes'],
                     transform_input=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(1, config['num_epochs'] + 1):
    train(epoch)

torch.save(model.state_dict(), 'model/cnn_dict.model')
torch.save(model, 'model/cnn.model')