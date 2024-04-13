import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np
import cv2
from .matlab_cp2tform import get_similarity_transform_for_cv2
import pandas as pd
import os
import sys
from scipy.spatial.distance import cdist
import argparse
import json
import math


def load_json2args(json_file):
	args = argparse.ArgumentParser().parse_args()
	with open(json_file, 'r') as f:
		args.__dict__ = json.load(f)
	return args


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False


def KFold(n=6000, n_folds=10, shuffle=False):
	folds = []
	base = list(range(n))
	for i in range(n_folds):
		test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
		train = list(set(base) - set(test))
		folds.append([train, test])
	return folds


def eval_acc(threshold, diff):
	y_predict = np.int32(diff[:, 0] > threshold)
	y_true = np.int32(diff[:, 1])
	accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
	return accuracy


def find_best_threshold(thresholds, predicts):
	best_threshold = best_acc = 0
	for threshold in thresholds:
		accuracy = eval_acc(threshold, predicts)
		if accuracy >= best_acc:
			best_acc = accuracy
			best_threshold = threshold
	return best_threshold


def save_model(model, filename):
	state = model.state_dict()
	for key in state: state[key] = state[key].clone().cpu()
	torch.save(state, filename)


def save_results(results, path):
	data_frame = pd.DataFrame(data=results)
	data_frame.to_csv(path + 'train_results.csv')


def alignment(src_img, src_pts, size=128):
	ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
	           [48.0252, 71.7366], [33.5493, 92.3655],
	           [62.7299, 92.2041]]
	if size is not None:
		ref_pts = np.array(ref_pts)
		ref_pts[:, 0] = ref_pts[:, 0] * size / 112
		ref_pts[:, 1] = ref_pts[:, 1] * size / 112
		crop_size = (size, size)
	else:
		crop_size = (96, 112)
	src_pts = np.array(src_pts).reshape(5, 2)
	s = np.array(src_pts).astype(np.float32)
	r = np.array(ref_pts).astype(np.float32)
	tfm = get_similarity_transform_for_cv2(s, r)
	face_img = cv2.warpAffine(src_img, tfm, crop_size, flags=cv2.INTER_CUBIC)
	# if size is not None:
	# 	face_img = cv2.resize(face_img, dsize=(96, 112), interpolation=cv2.INTER_CUBIC)
	return face_img


def alignment2(src_img, src_pts, scale=1.):
	output = (96, 112)
	warp_output = output
	center = (output[0]//2, output[1]//2)
	ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
	           [48.0252, 71.7366], [33.5493, 92.3655],
	           [62.7299, 92.2041]]
	src_pts = np.array(src_pts).reshape(5, 2)
	s = np.array(src_pts).astype(np.float32)
	r = np.array(ref_pts).astype(np.float32)
	tf_m = np.vstack((get_similarity_transform_for_cv2(s, r), np.asarray([[0, 0, 1]], dtype=np.float32)))
	aug_m = getAffineMtrx(center=center, angle=0., scale=scale)
	tf_m = np.dot(aug_m, tf_m)
	if scale != 1.:
		warp_output = tuple(int(np.rint(i * scale)) for i in warp_output)
	face_img = cv2.warpPerspective(src_img, tf_m, warp_output, flags=cv2.INTER_CUBIC, borderValue=0.0)
	if scale != 1.:
		face_img = cv2.resize(face_img, output, cv2.INTER_CUBIC)
	return face_img


def crop_align_resize(src_img, src_pts):
	src_pts = np.array(src_pts).reshape(5, 2)
	y_min, y_max = src_pts[:, 1].min(), src_pts[:, 1].max()
	size = int((y_max - y_min) * 3)
	crop_img = alignment(src_img, src_pts, size)
	return crop_img


class L2Norm(nn.Module):
	def forward(self, input, dim=1):
		return F.normalize(input, p=2, dim=dim)


def face_ToTensor(img):
    img = img.copy()  # Make a copy of the array
    img = img.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img).contiguous().float()
    img_tensor /= 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    return img_tensor


def save_log(losses, args):
	message = ''
	for name, value in losses.items():
		message += '{}: {:.4f} '.format(name, value)
	log_name = os.path.join(args.checkpoints_dir, 'log.txt')
	with open(log_name, "a") as log_file:
		log_file.write('\n' + message)


def tensor_pair_cosine_distance(features11, features12, features21, features22, type='normal'):
	if type == 'concat':
		features1 = torch.cat((features11, features12), dim=1)
		features2 = torch.cat((features21, features22), dim=1)
	elif type == 'sum':
		features1 = features11 + features12
		features2 = features21 + features22
	elif type == 'normal':
		features1 = features11
		features2 = features21
	else:
		print('tensor_pair_cosine_distance unspported type!')
		sys.exit()
	scores = torch.nn.CosineSimilarity()(features1, features2)
	scores = scores.cpu().numpy().reshape(-1, 1)
	return scores


def tensor_pair_cosine_distance_matrix(features11, features12, features21, features22, type='normal'):
	if type == 'concat':
		features1 = torch.cat((features11, features12), dim=1)
		features2 = torch.cat((features21, features22), dim=1)
	elif type == 'sum':
		features1 = features11 + features12
		features2 = features21 + features22
	elif type == 'normal':
		features1 = features11
		features2 = features21
	else:
		print('tensor_pair_cosine_distance_matrix unspported type!')
		sys.exit()
	features1_np = features1.cpu().numpy()
	features2_np = features2.cpu().numpy()
	scores = 1 - cdist(features2_np, features1_np, 'cosine')
	return scores

def getAffineMtrx(center=(96/2, 96/2), angle=45, translate=(0, 0), scale=1., shear=0):
	angle = math.radians(angle)
	shear = math.radians(shear)
	scale = 1.0 / scale

	# Inverted rotation matrix with scale and shear
	d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
	matrix = [
		math.cos(angle + shear), math.sin(angle + shear), 0,
		-math.sin(angle), math.cos(angle), 0,
		0, 0, 1
	]
	matrix = [scale / d * m for m in matrix]

	# Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
	matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
	matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

	# Apply center translation: C * RSS^-1 * C^-1 * T^-1
	matrix[2] += center[0]
	matrix[5] += center[1]

	return np.asarray(matrix, dtype=np.float32).reshape((3, 3))