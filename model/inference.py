from __future__ import print_function
import argparse

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import MultiStepLR


class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return TF.pad(image, padding, 0, 'constant')

class undercrop:
    def __call__(self, image):
        width, height = image.size
        if ( 2*width < height):
            h = int(0.1*height)
            i = int(0.9*height)
        elif (width < height):
            h = int(0.3*height)
            i = int(0.7*height)
        else:
            h = int(0.5 * height)
            i = int(0.5 * height)
        w = width
        j = 0
        return TF.crop(image, i, j, h, w)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--test_path', type=str,
    #                     help='path to test folder')
    args = parser.parse_args()
    # print (args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    transform=transforms.Compose([
        undercrop(),
        SquarePad(),
        transforms.GrayScale(output)
        transforms.Resize((512, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

    # Image OPEN
    test_image = transform(Image.open(args.test_path)).to(device).unsqueeze(dim=0)

    model = Net().to(device)
    checkpoint = torch.load('cnn_v5.pt')
    model.load_state_dict(checkpoint)
    model.eval()

    prediction = model(test_image).argmax()
    
    print (f'The prediction is: {prediction+1}')


if __name__ == '__main__':
    main()
