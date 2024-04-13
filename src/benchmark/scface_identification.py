import torch
from torch.utils.data import DataLoader
import os, cv2
import numpy as np
import pandas as pd
from sklearn import preprocessing
from common.util import tensor_pair_cosine_distance_matrix, face_ToTensor, alignment
from PIL import Image


_scface_root = '../../../Datasets/SCface/SCface_database/'
_scface_mugshot_landmarks = '../data/scface_mugshot.csv'
_scface_camera_landmarks = '../data/{}.csv'


class SCface_mugshot(torch.utils.data.Dataset):
	def __init__(self):
		super(SCface_mugshot, self).__init__()
		df = pd.read_csv(_scface_mugshot_landmarks, delimiter=",", header=None)
		numpyMatrix = df.values
		self.faces_path = numpyMatrix[:, 0]
		assert len(self.faces_path) == 130, "Wrong number of SCface mugshot"
		self.landmarks = numpyMatrix[:, 1:]
		self.targets, _ = self.get_targets()

	def __getitem__(self, index):
		face = cv2.imread(_scface_root + self.faces_path[index])
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		landmark = self.landmarks[index]
		face = alignment(face, landmark, size=128)
		face_flip = cv2.flip(face, 1)
		return face_ToTensor(face), face_ToTensor(face_flip), torch.LongTensor(self.targets[index])

	def __len__(self):
		return self.faces_path.shape[0]

	def get_targets(self):
		class_id = [path.split('/')[1].split('_')[0] for path in self.faces_path]
		unique_class_id = np.unique(class_id)
		le = preprocessing.LabelEncoder()
		le.fit(unique_class_id)
		targets = le.transform(class_id).reshape(-1, 1)
		num_class = unique_class_id.shape[0]
		return targets.astype(np.int32), num_class


class SCface_camera(torch.utils.data.Dataset):
	def __init__(self, name):
		super(SCface_camera, self).__init__()
		df = pd.read_csv(_scface_camera_landmarks.format(name), delimiter=",", header=None)
		numpyMatrix = df.values
		self.faces_path = numpyMatrix[:, 0]
		assert len(self.faces_path) == 650, "Wrong number of SCface {}".format(name)
		self.landmarks = numpyMatrix[:, 1:]
		self.targets, _ = self.get_targets()

	def __getitem__(self, index):
		face = cv2.imread(_scface_root + self.faces_path[index])
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		landmark = self.landmarks[index].reshape(-1, 2)
		y_min, y_max = landmark[:, 1].min(), landmark[:, 1].max()
		size = int((y_max - y_min) * 3)
		face = alignment(face, landmark, size=size)

		## Bicubic upsample
		face = Image.fromarray(face)
		face = face.resize((128, 128))
		face = np.asarray(face)

		face_flip = cv2.flip(face, 1)
		return face_ToTensor(face), face_ToTensor(face_flip), torch.LongTensor(self.targets[index])

	def __len__(self):
		return self.faces_path.shape[0]

	def get_targets(self):
		class_id = [path.split('/')[1].split('_')[0] for path in self.faces_path]
		unique_class_id = np.unique(class_id)
		le = preprocessing.LabelEncoder()
		le.fit(unique_class_id)
		targets = le.transform(class_id).reshape(-1, 1)
		num_class = unique_class_id.shape[0]
		return targets.astype(np.int32), num_class


def run(args, net, step=None):
	net.eval()
	dataloader = DataLoader(dataset=SCface_mugshot(), num_workers=2, batch_size=128,
	                        pin_memory=True, shuffle=False, drop_last=False)
	features_gallery1_total = []
	features_gallery2_total = []
	labels_gallery = []
	with torch.no_grad():
		bs_total = 0
		for index, (img1, img1_flip, targets) in enumerate(dataloader):
			bs = len(targets)
			img1, img1_flip = img1.to(args.device), img1_flip.to(args.device)
			features1, features2 = net(img1), net(img1_flip)
			features_gallery1_total += [features1.data]
			features_gallery2_total += [features2.data]
			labels_gallery += [targets.data]
			bs_total += bs
		assert bs_total == 130, '# of mugshot should be 130'
	features_gallery1_total = torch.cat(features_gallery1_total, 0)
	features_gallery2_total = torch.cat(features_gallery2_total, 0)
	labels_gallery = torch.cat(labels_gallery, 0)

	QUERIES = ['distance1', 'distance2', 'distance3']
	for query in QUERIES:
		dataset = SCface_camera('scface_{}'.format(query))
		dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=64,
		                        pin_memory=True, shuffle=False, drop_last=False)
		features_query1_total = []
		features_query2_total = []
		labels_query = []
		with torch.no_grad():
			bs_total = 0
			for index, (img1, img1_flip, targets) in enumerate(dataloader):
				bs = len(targets)
				img1, img1_flip = img1.to(args.device), img1_flip.to(args.device)
				features1, features2 = net(img1), net(img1_flip)
				features_query1_total += [features1.data]
				features_query2_total += [features2.data]
				labels_query += [targets.data]
				bs_total += bs
			assert bs_total == 650, '# of {} images should be 650'.format(query)
			features_query1_total = torch.cat(features_query1_total, 0)
			features_query2_total = torch.cat(features_query2_total, 0)
			labels_query = torch.cat(labels_query, 0)

		## Matching
		for cal_type in ['normal']:  # cal_type: concat/sum/normal
			scores_matrix = tensor_pair_cosine_distance_matrix(features_gallery1_total, features_gallery2_total,
			                                                   features_query1_total, features_query2_total,
			                                                   type=cal_type)
			predict_label = np.argmax(scores_matrix, axis=1)
			correct = predict_label == labels_query.cpu().numpy().reshape(-1)
			accuracy = correct.sum() / len(correct)
			if step is not None:
				message = 'SCface top-1 acc of {}: {} at {}iter (type: {})'.format(query, accuracy, step, cal_type)
			else:
				message = 'SCface top-1 acc of {}: {} at testing (type: {})'.format(query, accuracy, cal_type)
			print(message)
			if step is not None:
				log_name = os.path.join(args.checkpoints_dir, 'log.txt')
				with open(log_name, "a") as log_file:
					log_file.write('\n' + message)