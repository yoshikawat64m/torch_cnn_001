import torch
from torchviz import make_dot
from PIL import Image


def flatten(x):
    # (N, C, W, H)
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return x.view(-1, num_features)


def display(model, input_size=(227, 227), channel_size=3, batch_size=1):
    print(model)
    x = torch.zeros(batch_size, channel_size, input_size[0], input_size[1], dtype=torch.float, requires_grad=False)
    out = model.forward(x)
    dot = make_dot(out)
    dot.format = 'png'
    dot.render('graph_image')
    return Image.open('graph_image.png')
