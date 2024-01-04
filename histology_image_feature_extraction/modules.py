# -*- coding: utf-8 -*-
"""
Created on Nov 1 2023

@author: Haiyun Wang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torchvision import models


class simCLR_model(Module):
	def __init__(self, feature_dim=128):
		super(simCLR_model, self).__init__()

		self.f = []

		#load resnet50 structure
		for name, module in models.resnet50().named_children():
		# for name, module in res2net50().named_children():
			if name == 'conv1':
				module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)
		# projection head
		self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
							   nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

	def forward(self, x):
		x       = self.f(x)
		feature = torch.flatten(x, start_dim=1)
		out     = self.g(feature)

		return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class resnet50_model(Module):
	def __init__(self):
		super(resnet50_model, self).__init__()

		### load pretrained resnet50 model
		resnet50 = models.resnet50(pretrained=True)

		for param in resnet50.parameters():
			param.requires_grad = False

		self.f = []

		for name, module in resnet50.named_children():
			if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)

	def forward(self, x):
		x       = self.f(x)
		feature = torch.flatten(x, start_dim=1)
	   
		return F.normalize(feature, dim=-1)

