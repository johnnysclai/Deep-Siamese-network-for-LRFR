import torch
from torch.utils.data import DataLoader
import os, cv2, copy
import pandas as pd
import numpy as np
from common.util import alignment, face_ToTensor
from common.util import KFold, find_best_threshold, eval_acc, tensor_pair_cosine_distance
from PIL import Image


_lfw_root = '../../../Datasets/LFW/lfw/'
_lfw_landmarks = '../data/LFW.csv'
_lfw_pairs = '../data/lfw_pairs.txt'


class LFWDataset(torch.utils.data.Dataset):
	def __init__(self, img2_size):
		super(LFWDataset, self).__init__()
		df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
		numpyMatrix = df.values
		self.landmarks = numpyMatrix[:, 1:]
		self.df = df
		with open(_lfw_pairs) as f:
			pairs_lines = f.readlines()[1:]
		self.pairs_lines = pairs_lines
		self.img2_size = img2_size

	def __getitem__(self, index):
		p = self.pairs_lines[index].replace('\n', '').split('\t')
		if 3 == len(p):
			sameflag = np.int32(1).reshape(1)
			name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
			name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
		if 4 == len(p):
			sameflag = np.int32(0).reshape(1)
			name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
			name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
		img1 = cv2.imread(_lfw_root + name1)
		img2 = cv2.imread(_lfw_root + name2)
		facial5points1 = self.landmarks[self.df.loc[self.df[0] == name1].index.values[0]].reshape(5, 2)
		facial5points2 = self.landmarks[self.df.loc[self.df[0] == name2].index.values[0]].reshape(5, 2)
		img1 = alignment(img1, facial5points1, size=128)
		img2 = alignment(img2, facial5points2, size=128)
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

		## img2 downsample
		if self.img2_size < 128:
			img2 = Image.fromarray(img2)
			img2 = img2.resize((self.img2_size, self.img2_size), )
			img2 = img2.resize((128, 128))
			img2 = np.asarray(img2)

		img1_flip = cv2.flip(img1, 0)
		img2_flip = cv2.flip(img2, 0)
		params_dict = {
			'img1': face_ToTensor(img1),
			'img2': face_ToTensor(img2),
			'img1_flip': face_ToTensor(img1_flip),
			'img2_flip': face_ToTensor(img2_flip),
			'label': torch.LongTensor(sameflag)
		}
		return params_dict

	def __len__(self):
		return len(self.pairs_lines)


class LFW():
	def __init__(self, args=None, bs=64, workers=12, img2_size=128, writer=None):
		self.dataloader = DataLoader(dataset=LFWDataset(img2_size),
		                             num_workers=workers,
		                             batch_size=bs,
		                             pin_memory=True,
		                             shuffle=False,
		                             drop_last=False)
		self.args = copy.deepcopy(args)
		self.device = self.args.device
		self.writer = writer
		self.img2_size = img2_size

	def run(self, net, step=None):
		net.eval()
		features11 = []
		features21 = []
		features12 = []
		features22 = []
		labels = []
		with torch.no_grad():
			bs_total = 0
			for index, data in enumerate(self.dataloader):
				img1 = data['img1'].to(self.device)
				img2 = data['img2'].to(self.device)
				img1_flip = data['img1_flip'].to(self.device)
				img2_flip = data['img2_flip'].to(self.device)
				label = data['label']
				features11 += [net(img1).data]
				features21 += [net(img2).data]
				features12 += [net(img1_flip).data]
				features22 += [net(img2_flip).data]
				labels += [label.data]
				bs_total += len(label)
			assert bs_total == 6000, print('LFW pairs should be 6,000!')
		features11 = torch.cat(features11, 0)
		features21 = torch.cat(features21, 0)
		features12 = torch.cat(features12, 0)
		features22 = torch.cat(features22, 0)
		labels = torch.cat(labels, 0).numpy()
		for cal_type in ['normal']:  # cal_type: concat/sum/normal
			scores = tensor_pair_cosine_distance(features11, features12, features21, features22, type=cal_type)
			accuracy = []
			thd = []
			folds = KFold(n=6000, n_folds=10, shuffle=False)
			thresholds = np.linspace(-10000, 10000, 10000 + 1)
			thresholds = thresholds / 10000
			predicts = np.hstack((scores, labels))
			for idx, (train, test) in enumerate(folds):
				best_thresh = find_best_threshold(thresholds, predicts[train])
				accuracy.append(eval_acc(best_thresh, predicts[test]))
				thd.append(best_thresh)
			mean_acc, std = np.mean(accuracy), np.std(accuracy)
			writer_name = 'LFW_{}_{}'.format(cal_type, self.img2_size)
			if step is not None:
				message = '{} {:.4f} std={:.4f} at {}iter.'.format(writer_name, mean_acc, std, step)
			else:
				message = '{} {:.4f} std={:.4f} at testing.'.format(writer_name, mean_acc, std)
			message += '({}x{})'.format(self.img2_size, self.img2_size)
			print(message)
			if step is not None:
				log_name = os.path.join(self.args.checkpoints_dir, 'log.txt')
				with open(log_name, "a") as log_file:
					log_file.write('\n' + message)
			if self.writer is not None:
				self.writer.add_scalar(writer_name, mean_acc, step)
