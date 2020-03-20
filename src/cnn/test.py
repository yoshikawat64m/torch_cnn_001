import torch
from torch.utils.data import DataLoader
from cnn_finetune import make_model
from sklearn.metrics import classification_report
import os
from MyDataset import MyDataset
from inception import inception_v3


def test():
    model.eval()
    pred = []
    Y = []
    for i, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            output = model(x)
        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in y]

    print(classification_report(Y, pred))


config = {
    'dataset_dir': 'dataset/flower_images/',
    'label_file': 'dataset/flower_images/flower_labels.csv',
    'num_classes': 10,
    'batch_size': 70,
    'input_size': 299,
}

test_set = MyDataset(config['label_file'], config['dataset_dir'],  size=config['input_size'])
test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=True)

model = inception_v3(num_classes=config['num_classes'],
                     transform_input=True)

model.load_state_dict(torch.load('model/cnn_dict.model'))

test()