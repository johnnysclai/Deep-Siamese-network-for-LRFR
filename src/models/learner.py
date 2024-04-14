import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

torch.backends.cudnn.benchmark = True

import os, copy, math
import numpy as np
from collections import OrderedDict
from . import networks, losses, heads
from backbone import model_irse, model_resnet
from torch.nn.parallel import DistributedDataParallel as DDP

class CreateLearner(nn.Module):
	def __init__(self, args):
		super(CreateLearner, self).__init__()
		self.args = copy.deepcopy(args)
		self.feature_dim = args.feature_dim
		self.device = args.device
		self.gpu_ids = args.gpu_ids
		if 'spherenet' in args.backbone:
			num_layers = int(args.backbone.split('spherenet')[-1])
			self.backbone = getattr(networks, 'spherenet')(num_layers, args.feature_dim, args.use_pool,
			                                               args.use_dropout)
		elif 'ResNet' in args.backbone:
			self.backbone = getattr(model_resnet, args.backbone)((128, 128))
		elif 'IR' in args.backbone:
			self.backbone = getattr(model_irse, args.backbone)((128, 128))
		else:
			raise NotImplementedError('Backbone model [{:s}] is not found'.format(args.backbone))
		self.backbone.to(self.device)

	def train_setup(self, class_num):
		self.model_names = ['backbone', 'head']
		self.loss_names = ['loss', 'loss_cls', 'lr']
		if self.args.multi:
			self.loss_names += ['loss_cls_lr1', 'loss_cls_lr2', 'loss_cls_lr3']
		if self.args.lambda_dist > 0.:
			self.loss_names += ['loss_dist']
		self.losses = {
			'celoss': nn.CrossEntropyLoss(),
			'focalloss': losses.FocalLoss(self.args.gamma)
		}

		## Create head
		self.head = getattr(heads, self.args.head)
		self.head = self.head(self.args, class_num)
		self.head.to(self.device)

		## Create loss function
		self.criterion = self.losses[self.args.loss]

		## Setup optimizer
		self.lr = self.args.lr
		self.save_dir = self.args.checkpoints_dir

		params = list(self.backbone.parameters()) + list(self.head.parameters())
		self.optimizer = optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
		self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.decay_steps, gamma=0.1)

		## Setup nn.DataParallel if necessary
		if len(self.gpu_ids) > 1:
			self.backbone = nn.DataParallel(self.backbone)

		## Switch to training mode
		self.train()
		self.steps = 0

		if self.args.lambda_dist > 0.:
			self.pos_mask, self.neg_mask = self.get_mask()

	def weights_init(self, m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			torch.nn.init.xavier_uniform_(m.weight.data)
			m.bias.data.fill_(0)
		elif classname.find('Linear') != -1:
			torch.nn.init.xavier_uniform_(m.weight.data)
			m.bias.data.fill_(0)

	def update_learning_rate(self):
		self.scheduler.step()
		self.lr = self.optimizer.param_groups[0]['lr']

	def optimize_parameters(self, data):
		self.target = data['target'].view(-1).to(self.device)
		self.face_hr = data['face_hr'].to(self.device)
		self.face_lr1 = data['face_8'].to(self.device)
		self.face_lr2 = data['face_12'].to(self.device)
		self.face_lr3 = data['face_16'].to(self.device)
		self.face = self.face_hr

		if self.args.multi:
			self.feature_hr = self.backbone(self.face_hr)
			self.feature_lr1 = self.backbone(self.face_lr1)
			self.feature_lr2 = self.backbone(self.face_lr2)
			self.feature_lr3 = self.backbone(self.face_lr3)
		else:
			self.feature = self.backbone(self.face)

		## Forward G
		self.forward()
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

	def get_current_losses(self):
		errors_ret = OrderedDict()
		for name in self.loss_names:
			if isinstance(name, str):
				# float(...) works for both scalar tensor and float number
				errors_ret[name] = float(getattr(self, name))
		return errors_ret

	def cosine_simililarity(self, x, y):
		v = torch.nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0))
		return v

	def get_mask(self):
		diag = np.eye(self.args.bs)
		mask = np.hstack((diag, diag))
		mask = np.hstack((mask, mask))
		mask = np.vstack((mask, mask))
		mask = np.vstack((mask, mask))
		negative_mask = torch.from_numpy((1 - mask)).contiguous().type(torch.bool).to(self.device)
		l = np.eye(self.args.bs * 4)
		positive_mask = torch.from_numpy((mask - l)).contiguous().type(torch.bool).to(self.device)
		return positive_mask, negative_mask

	def save_networks(self, which_epoch):
		for name in self.model_names:
			if isinstance(name, str):
				save_filename = '{:07d}_net_{}.pth'.format(which_epoch, name)
				save_path = os.path.join(self.save_dir, save_filename)
				net = getattr(self, name)
				model_state_dict = OrderedDict()
				if len(self.gpu_ids) > 1 and torch.cuda.is_available():
					try:
						state_dict = net.module.state_dict()
					except:
						state_dict = net.state_dict()
				else:
					state_dict = net.state_dict()
				for k, v in state_dict.items():
					model_state_dict[k] = v.to('cpu')
				torch.save(model_state_dict, save_path)

	def forward(self, is_feature=False):
		if is_feature or self.target is None:
			return self.features
		else:
			self.loss = 0.
			if self.args.multi:
				output_dict = self.head(self.feature_hr, self.target.view(-1, 1))
				self.loss_cls = self.criterion(output_dict['logit'], self.target)
				output_dict_lr1 = self.head(self.feature_lr1, self.target.view(-1, 1))
				self.loss_cls_lr1 = self.criterion(output_dict_lr1['logit'], self.target)
				output_dict_lr2 = self.head(self.feature_lr2, self.target.view(-1, 1))
				self.loss_cls_lr2 = self.criterion(output_dict_lr2['logit'], self.target)
				output_dict_lr3 = self.head(self.feature_lr3, self.target.view(-1, 1))
				self.loss_cls_lr3 = self.criterion(output_dict_lr3['logit'], self.target)
				self.loss += (self.loss_cls + self.loss_cls_lr1 + self.loss_cls_lr2 + self.loss_cls_lr3) / 4
			else:
				output_dict = self.head(self.feature, self.target.view(-1, 1))
				self.loss_cls = self.criterion(output_dict['logit'], self.target)
				self.loss = self.loss_cls

			if self.args.lambda_dist > 0.:
				representations = torch.cat((self.feature_hr, self.feature_lr1, self.feature_lr2, self.feature_lr3))
				dist_matrix = 1 - self.cosine_simililarity(representations, representations)
				pos = dist_matrix[self.pos_mask].view(len(dist_matrix), -1)
				neg = dist_matrix[self.neg_mask].view(len(dist_matrix), -1)
				hard_pos = torch.max(pos, 1)[0]
				hard_neg = torch.min(neg, 1)[0]
				loss = torch.clamp(1.0 + hard_pos - hard_neg, min=0.0)
				self.loss_dist = loss.mean()
				self.loss += self.args.lambda_dist * self.loss_dist
		self.steps += 1

	def train_disable(self):
		for name in self.model_names:
			try:
				if isinstance(name, str):
					getattr(self, name).eval()
			except:
				print('{}.eval() cannot be implemented as {} does not exist.'.format(name, name))

	def train_enable(self):
		for name in self.model_names:
			try:
				if isinstance(name, str):
					getattr(self, name).train()
			except:
				print('{}.train() cannot be implemented as {} does not exist.'.format(name, name))

	def set_requires_grad(self, nets, requires_grad=False):
		"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
		Parameters:
			nets (network list)   -- a list of networks
			requires_grad (bool)  -- whether the networks require gradients or not
		"""
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def get_target_tensor(self, prediction, target_is_real):
		"""Create label tensors with the same size as the input.

		Parameters:
			prediction (tensor) - - tpyically the prediction from a discriminator
			target_is_real (bool) - - if the ground truth label is for real images or fake images

		Returns:
			A label tensor filled with ground truth label, and with the size of the input
		"""

		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction)
