import torch
from cnn_finetune import make_model
from sklearn.metrics import classification_report
import os
from MyDataset import MyDataset


def test():
    model.eval()
    pred = []
    Y = []
    for i, (x,y) in enumerate(test_loader):
        with torch.no_grad():
            output = model(x)
        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in y]

    print(classification_report(Y, pred))


dataset_dir = 'dataset/flower_images/'
label_file= 'dataset/flower_images/flower_labels.csv'

test_set = MyDataset(label_file, dataset_dir)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

model = make_model('vgg16', num_classes=10, input_size=(224, 224))
param = torch.load('model/cnn_dict.model')
model.load_state_dict(param)

test()