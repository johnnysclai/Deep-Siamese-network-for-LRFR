import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.util import L2Norm


class softmax(nn.Module):
	def __init__(self, args, class_num, bias=False):
		super(softmax, self).__init__()
		self.in_features = args.feature_dim
		self.out_features = class_num
		self.s = args.s
		self.weights = torch.nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
		if bias:
			self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_features))
		else:
			self.register_parameter('bias', None)
		nn.init.xavier_uniform_(self.weights)
		if self.bias is not None:
			nn.init.zero_(self.bias)

	def forward(self, input, target):
		scores = F.linear(input, self.weights, self.bias)  # x @ weight.t() + bias(if any)
		feed_dict = {
			'logit': scores
		}
		return feed_dict


## Not ready yet
class asoftmax(nn.Module):
	def __init__(self, args, class_num):
		super(asoftmax, self).__init__()
		self.in_features = args.feature_dim
		self.out_features = class_num
		self.m = args.m_1  # defalut m = 4
		self.iter = 0
		self.base = 1000.0
		self.gamma = 0.12
		self.power = 1
		self.LambdaMin = 5.0
		self.weights = torch.nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
		self.weights.data.uniform_(-1, 1)  # weightc initialization
		# self.weights.data.uniform_(-1, 1).renorm_(p=2, dim=1, maxnorm=1e-5).mul_(1e5)  # weight initialization
		assert self.m >= 1., 'margin m of asoftmax should >= 1.0'

		# duplication formula
		self.mlambda = [
			lambda x: x ** 0,
			lambda x: x ** 1,
			lambda x: 2 * x ** 2 - 1,
			lambda x: 4 * x ** 3 - 3 * x,
			lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
			lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
		]

	def forward(self, input, target):
		# self.it += 1
		# self.Lambda = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
		# scores = F.linear(input, F.normalize(self.weights))  # x @ weight.t() + bias(if any)
		# index = torch.zeros_like(scores).scatter_(1, target, 1)
		# x_l2norm = input.norm(dim=1)
		# cos_theta = scores / (x_l2norm.view(-1, 1).clamp(min=1e-12))
		# cos_theta = cos_theta.clamp(-1, 1)
		# m_theta = self.m * cos_theta.data.acos()
		# k = (m_theta / 3.141592653589793).floor()
		# cos_m_theta = torch.cos(m_theta)
		# phi_theta = ((-1) ** k) * cos_m_theta - 2 * k
		# phi_theta = phi_theta * x_l2norm.view(-1, 1)
		# scores_new = scores - (scores * index / (1 + self.Lambda)) + (phi_theta * index / (1 + self.Lambda))

		self.iter += 1
		self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
		cos_theta = F.linear(F.normalize(input), F.normalize(self.weights))
		cos_theta = cos_theta.clamp(-1, 1)
		cos_m_theta = self.mlambda[self.m](cos_theta)
		theta = cos_theta.data.acos()
		k = (self.m * theta / 3.14159265).floor()
		phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
		NormOfFeature = torch.norm(input, 2, 1)

		# --------------------------- convert label to one-hot ---------------------------
		one_hot = torch.zeros_like(cos_theta)
		one_hot.scatter_(1, target.view(-1, 1), 1)

		# --------------------------- Calculate output ---------------------------
		output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
		output *= NormOfFeature.view(-1, 1)

		feed_dict = {
			'logit': output
		}
		return feed_dict


class amsoftmax(nn.Module):
	def __init__(self, args, class_num):
		super(amsoftmax, self).__init__()
		self.args = args
		self.in_features = args.feature_dim
		self.out_features = class_num
		self.s = args.s  # default: 30(AMSoftmax)/64(CosFace)
		self.m = args.m_3  # default: 0.35
		self.weights = torch.nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
		# self.weights.data.uniform_(-1, 1).renorm_(p=2, dim=1, maxnorm=1e-5).mul_(1e5)  # weight initialization
		self.weights.data.uniform_(-1, 1)  # weightc initialization
		assert self.s > 1.0, 'scaling factor s should > 1.0'
		assert self.m > 0., 'scaling factor s should > 1.0'

	def forward(self, input, target):
		cos_theta = F.linear(F.normalize(input, p=2), F.normalize(self.weights, p=2))  # x @ weight.t() + bias(if any)
		cos_theta = cos_theta.clamp(min=-1, max=1)
		mask = torch.zeros_like(cos_theta).scatter_(1, target, 1)
		output = cos_theta * (1 - mask) + (cos_theta - self.m) * mask
		output *= self.s
		feed_dict = {
			'logit': output
		}
		return feed_dict


class centerloss(nn.Module):
	def __init__(self, class_num, args, bias=False):
		super(centerloss, self).__init__()
		self.device = args.device
		self.lamb = args.lamb  # weight of center loss
		self.alpha = 0.5  # weight of updating centers
		self.in_features = args.feature_dim
		self.class_num = class_num
		self.f_norm = args.use_f_norm
		self.centers = torch.nn.Parameter(torch.Tensor(self.class_num, self.in_features))
		self.centers.requires_grad = False
		self.delta_centers = torch.zeros_like(self.centers)
		self.softmaxloss = softmax(class_num, args)
		self.reset_parameters()

	def forward(self, input, target):
		scores, loss = self.softmaxloss(input, target)  # Softmax loss
		'''
			Center loss: follow the paper's implementation.
			Inspired by https://github.com/louis-she/center-loss.pytorch/blob/5be899d1f622d24d7de0039dc50b54ce5a6b1151/loss.py
		'''
		## Center loss
		if self.f_norm:
			x = L2Norm()(input)
		else:
			x = input
		self.update_center(x, target)

		target_centers = self.centers[target].squeeze()
		center_loss = ((x - target_centers) ** 2).sum(dim=1).mean()
		return scores, loss + self.lamb * 0.5 * center_loss

	def update_center(self, features, targets):
		# implementation equation (4) in the center-loss paper
		targets, indices = torch.sort(targets.view(-1))
		target_centers = self.centers[targets]
		features = features.detach()[indices]
		delta_centers = target_centers - features
		uni_targets, indices = torch.unique(targets.cpu(), sorted=True, return_inverse=True)
		uni_targets = uni_targets.to(self.device)
		indices = indices.to(self.device)
		delta_centers = torch.zeros(uni_targets.size(0), delta_centers.size(1)).to(self.device).index_add_(0, indices,
		                                                                                                   delta_centers)
		targets_repeat_num = uni_targets.size()[0]
		uni_targets_repeat_num = targets.size()[0]
		targets_repeat = targets.repeat(targets_repeat_num).view(targets_repeat_num, -1)
		uni_targets_repeat = uni_targets.unsqueeze(1).repeat(1, uni_targets_repeat_num)
		same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)
		delta_centers = delta_centers / (same_class_feature_count + 1.0) * self.alpha
		result = torch.zeros_like(self.centers)
		result[uni_targets, :] = delta_centers
		self.centers -= result

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))


class adacos(nn.Module):
	def __init__(self, args, class_num, fix=False):
		super(adacos, self).__init__()
		self.args = args
		self.in_features = args.feature_dim
		self.class_num = class_num
		self.weights = torch.nn.Parameter(torch.FloatTensor(self.class_num, self.in_features))
		# nn.init.xavier_uniform_(self.weights)
		# self.weights.data.uniform_(-1, 1).renorm_(p=2, dim=1, maxnorm=1e-5).mul_(1e5)  # weight initialization
		self.weights.data.uniform_(-1, 1)  # weightc initialization
		self.fix = fix
		self.theta_o = math.pi / 180 * 45  # 45 by default
		self.s = math.log(self.class_num - 1) / self.theta_o
		self.iter = 0

	def forward(self, input, target):
		x = F.normalize(input, p=2)
		weights = F.normalize(self.weights, p=2)
		cos_theta = F.linear(x, weights)  # x @ weight.t() + bias(if any)
		cos_theta = cos_theta.clamp(min=-1, max=1)
		## Dyna. AdaCos
		mask = torch.zeros_like(cos_theta).scatter_(1, target, 1)
		# Calculate B_avg, Eq.(13)
		B_avg = torch.where(mask < 1, torch.exp(self.s * cos_theta), torch.zeros_like(cos_theta)).data
		B_avg = B_avg.sum() / len(input)
		# Calculate cos_theta_med
		theta = torch.acos(cos_theta).data
		theta_i = theta[mask == 1]
		theta_k = theta[mask != 1]
		theta_med = torch.median(theta_i)
		# Calculate s_d, Eq. (15)
		self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_o * torch.ones_like(theta_med), theta_med))

		output = self.s * cos_theta
		feed_dict = {
			'logit': output,
			'theta': theta_med,
			'theta_k': theta_k.mean()
		}
		self.iter += 1
		return feed_dict


class fixadacos(nn.Module):
	def __init__(self, args, class_num, fix=True):
		super(fixadacos, self).__init__()
		self.args = args
		self.in_features = args.feature_dim
		self.class_num = class_num
		self.weights = torch.nn.Parameter(torch.FloatTensor(self.class_num, self.in_features))
		nn.init.xavier_uniform_(self.weights)
		# self.weights.data.uniform_(-1, 1).renorm_(p=2, dim=1, maxnorm=1e-5).mul_(1e5)  # weight initialization
		self.fix = fix
		self.s = math.sqrt(2) * math.log(self.class_num - 1)

	def forward(self, input, target):
		x = F.normalize(input, p=2)
		weights = F.normalize(self.weights, p=2)
		cos_theta = F.linear(x, weights)  # x @ weight.t() + bias(if any)
		cos_theta = cos_theta.clamp(min=-1, max=1)

		mask = torch.zeros_like(cos_theta).scatter_(1, target, 1)
		# Calculate cos_theta_med
		theta = torch.acos(cos_theta).data
		theta_med = torch.median(theta[mask == 1])
		theta_k = theta[mask == 0]

		## Dyna. AdaCos
		if not self.fix and self.iter > 0:
			# Calculate B_avg, Eq.(13)
			B_avg = torch.where(mask < 1, torch.exp(self.s * cos_theta), torch.zeros_like(cos_theta)).data
			B_avg = B_avg.sum() / len(input)
			# Calculate s_d, Eq. (15)
			self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))

		output = self.s * cos_theta
		feed_dict = {
			'logit': output,
			'theta': theta_med,
			'theta_k': theta_k.mean()
		}
		return feed_dict


class sv_softmax(nn.Module):
	def __init__(self, args, class_num, fix=True):
		super(sv_softmax, self).__init__()
		self.args = args
		self.in_features = args.feature_dim
		self.class_num = class_num
		self.weights = torch.nn.Parameter(torch.FloatTensor(self.class_num, self.in_features))
		nn.init.xavier_uniform_(self.weights)
		# self.weights.data.uniform_(-1, 1).renorm_(p=2, dim=1, maxnorm=1e-5).mul_(1e5)  # weight initialization
		self.fix = fix
		self.s = math.sqrt(2) * math.log(self.class_num - 1)
		self.t = 1.2

	def forward(self, input, target):
		x = F.normalize(input, p=2)
		weights = F.normalize(self.weights, p=2)
		cos_theta = F.linear(x, weights)  # x @ weight.t() + bias(if any)
		cos_theta = cos_theta.clamp(min=-1, max=1)

		mask = torch.zeros_like(cos_theta).scatter_(1, target, 1)

		cos_theta_i = cos_theta[mask == 1].data
		binaryMask = (cos_theta.data > cos_theta_i.view(-1, 1)).to(self.args.device, dtype=torch.float32)
		binaryMask = binaryMask * (1-mask)

		# Calculate cos_theta_med
		theta = torch.acos(cos_theta).data
		theta_med = torch.median(theta[mask == 1])
		theta_k = theta[mask == 0]

		# output = self.s * cos_theta

		output = (1-binaryMask) * cos_theta + binaryMask * (cos_theta + (self.t-1)*(cos_theta+1))
		output *= self.s

		feed_dict = {
			'logit': output,
			'theta': theta_med,
			'theta_k': theta_k.mean()
		}
		return feed_dict
