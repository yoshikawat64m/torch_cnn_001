
import torch
import torch.nn as nn
import torch.optim as optim

import datetime

from dataset import MyDataset

from models.inception.inception_v3 import inception_v3


class TrainModel:

    def __init__(self):
        config = {
            'dataset_dir': 'dataset/flower_images/',
            'label_file': 'dataset/flower_images/flower_labels.csv',
            'num_classes': 10,
            'batch_size': 70,
            'num_epochs': 30,
            'pretrained': False,
            'input_size': 299,
        }

        dataset = MyDataset(
            config['label_file'],
            config['dataset_dir'],
            size=config['input_size']
        )
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = inception_v3(
            pretrained=config['pretrained'],
            num_classes=config['num_classes'],
            transform_input=True
        ).to(self.device)

    def run_train(self, num_epochs):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

        for epoch in range(1, num_epochs + 1):
            self._train(epoch)

        torch.save(self.model.state_dict(), 'model/cnn_dict.model')
        torch.save(self.model, 'model/cnn.model')

    def _train(self, epoch):
        total_loss = 0
        total_size = 0

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output[0], target)

            total_loss += loss.item()
            total_size += data.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 1000 == 0:
                now = datetime.datetime.now()
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                    now,
                    epoch,
                    batch_idx * len(self.data),
                    len(self.data_loader.dataset),
                    100 * batch_idx / len(self.data_loader),
                    total_loss / total_size
                ))
