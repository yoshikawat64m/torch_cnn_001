
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_finetune import make_model
from keras.datasets import fashion_mnist

import datetime
import matplotlib.pyplot as plt
import os

from MyDataset import MyDataset


def train(epoch):
    total_loss = 0
    total_size = 0
    import pdb; pdb.set_trace()
    
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
                100. * batch_idx / len(train_loader), total_loss / total_size))


num_epochs = 100
batch_size = 32

dataset_dir = 'dataset/flower_images/'
label_file= 'dataset/flower_images/flower_labels.csv'

save_model_path = 'model/cnn.model'

model_name = 'vgg16'
num_classes = 10

run_type = 'cpu'

train_set = MyDataset(label_file, dataset_dir)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

model = make_model(model_name, num_classes=num_classes, pretrained=True, input_size=(224, 224))

device = torch.device(run_type)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, num_epochs + 1):
    train(epoch)

#torch.save(model.state_dict(), 'model/cnn_dict.model')
torch.save(model, save_model_path)