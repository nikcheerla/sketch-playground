# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import sys, random, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from scipy.ndimage import imread

from PIL import Image
import matplotlib.pyplot as plt

import IPython

import torchvision.transforms as transforms
import torchvision.models as models
from scipy.ndimage import filters

import copy


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

imsize = 256  # use small size if no gpu

loader = transforms.Compose([
    transforms.Scale(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

print ("Made loader")
def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image)[0:3, 0:imsize, 0:imsize])

    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


print ("Loading style/content images")

style_img = image_loader(sys.argv[1]).type(dtype)
content_img = image_loader(sys.argv[2]).type(dtype)

print ("Loaded style/content images")

#assert style_img.size() == content_img.size(), \
#    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, file_name=None):
    print ("Saving to: ", file_name)
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    plt.imsave(file_name, image)

imshow(style_img.data, file_name='style.jpg')
imshow(content_img.data, file_name='content.jpg')


class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kernel = filters.gaussian_filter(kernel, sigma=1.4)
        self.gaussian = torch.Tensor(kernel).view(1, 1, 5, 5)
        self.gaussian = self.gaussian.repeat(target.size(1), target.size(1), 1, 1)

        self.sobelx = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)
        self.sobely = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        self.sobelx = self.sobelx.repeat(target.size(1), target.size(1), 1, 1)
        self.sobely = self.sobely.repeat(target.size(1), target.size(1), 1, 1)

        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        #print (input.max())

        #x = F.conv2d(input, weight=Variable(self.gaussian), padding=2)
        x = input
        Gx = F.conv2d(x, weight=Variable(self.sobelx), padding=1)
        Gy = F.conv2d(x, weight=Variable(self.sobely), padding=1)
        G = torch.sqrt(Gx*Gx + Gy*Gy)
        #G = F.conv2d(G, weight=Variable(self.gaussian), padding=2)
        
        Gn = (G - G.mean())/(G.std() + 1e-5)
        tn = (self.target - self.target.mean())/(self.target.std() + 1e-5)

        #print (G.max())

        self.loss = 10*self.criterion(Gn * self.weight, -tn * self.weight)
        self.loss += 2*self.criterion(input, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss



class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)




class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss



cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()

content_layers_default = ['conv_3', 'conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses



input_img = content_img.clone()

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer



def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=550,
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    return input_param.data




output = run_style_transfer(cnn, content_img, style_img, input_img)
output -= (1.0-content_img).data
output = output.clamp(0, 1)

imshow(output, file_name='out.jpg')
