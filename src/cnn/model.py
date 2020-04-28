
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import datetime

from dataset import MyDataset

from models import *


class Config:
    image_dir = 'datasets/flower_images/'
    train_label = 'datasets/flower_images/flower_labels.csv'
    test_label = 'datasets/flower_images/flower_labels.csv'

    model = 'inception_v3'
    model_state = 'models/inception/model/inception_v3.model'
    model_state_dict = 'model/inception/model/inception_v3_dict.model'
    input_size = 299
    num_classes = 10
    batch_size = 50
    num_epochs = 30
    per_out = 100

    pretrained = False


class Model:

    def __init__(self, mode='train'):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.config = Config()

        if mode == 'train':
            dataset = MyDataset(
                label_file=self.config.train_label,
                image_dir=self.config.image_dir,
                size=self.config.input_size
            )
        else:
            dataset = MyDataset(
                label_file=self.config.test_label,
                image_dir=self.config.image_dir,
                size=self.config.input_size
            )

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.model = eval(self.config.model)(
            pretrained=self.config.pretrained,
            num_classes=self.config.num_classes,
            transform_input=True
        ).to(self.device)

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

    def train(self):
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.config.num_epochs + 1):
            self.total_loss = 0
            self.total_size = 0

            for batch_i, (data, target) in enumerate(self.data_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output[0], target)

                self.total_loss += loss.item()
                self.total_size += data.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_i % self.config.per_out == 0:
                    self._output_train_result(epoch, batch_i)

        torch.save(self.model, self.config.model_state)
        torch.save(self.model.state_dict(), self.config.model_state_dict)

    def test(self):
        self.model.load_state_dict(torch.load(self.config.model_state_dict))

        predictions = []
        answers = []

        for batch_i, (data, target) in enumerate(self.data_loader):
            with torch.no_grad():
                output = self.model(data)
            predictions += [int(l.argmax()) for l in output]
            answers += [int(l) for l in target]

        print(classification_report(answers, predictions))

    def _output_train_result(self, epoch, batch_i):
        now = datetime.datetime.now()
        print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
            now,
            epoch,
            self.total_size,
            len(self.data_loader.dataset),
            100 * batch_i / len(self.data_loader),
            self.total_loss / self.total_size
        ))
