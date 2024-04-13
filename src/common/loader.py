import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
import os, cv2, random
import numpy as np
import pandas as pd
from glob import glob
from sklearn import preprocessing
import scipy.io as sio
from .util import alignment, crop_align_resize, face_ToTensor
from .util import alignment2

_vggface2_root = '../../../Datasets/VGGFace2/train_test_128x128/'

_survface_root = '../../../Datasets/QMUL-SurvFace/QMUL-SurvFace/'
_scface_root = '../../../Datasets/SCface/SCface_database/'
_scface_mugshot_landmarks = '../data/scface_mugshot.csv'
_scface_camera_landmarks = '../data/{}.csv'

_lfw_root = '../../../Datasets/LFW/lfw/'
_lfw_landmarks = '../data/LFW.csv'
_lfw_pairs = '../data/lfw_pairs.txt'
_lfw_labels = '../data/lfw_labels.txt'

class VGGFace2Dataset(torch.utils.data.Dataset):
    def __init__(self, is_aug=True):
        super(VGGFace2Dataset, self).__init__()
        self.faces_path = glob(f'{_vggface2_root}/*/*.jpg')
        self.faces_path.sort()
        self.targets, self.num_class = self.get_targets()
        t = self.targets[0][0]
        self.subject_total = []
        list = []
        for index, target in enumerate(self.targets):
            if target[0] == t:
                list += [index]
            elif target[0] != t:
                self.subject_total += [list]
                list = []
                list += [index]
                t = target[0]
            if index == len(self.faces_path)-1:
                self.subject_total += [list]
        self.is_aug = is_aug

    def __getitem__(self, index):
        list = self.subject_total[index]
        index8 = random.randint(list[0],list[-1])
        index12 = random.randint(list[0],list[-1])
        index16 = random.randint(list[0],list[-1])
        index = random.randint(list[0],list[-1])

        face8 = cv2.imread(self.faces_path[index8])
        face8 = cv2.cvtColor(face8, cv2.COLOR_BGR2RGB)
        face12 = cv2.imread(self.faces_path[index12])
        face12 = cv2.cvtColor(face12, cv2.COLOR_BGR2RGB)
        face16 = cv2.imread(self.faces_path[index16])
        face16 = cv2.cvtColor(face16, cv2.COLOR_BGR2RGB)
        face = cv2.imread(self.faces_path[index])
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if self.is_aug:
            if random.random() > 0.5:
                face = cv2.flip(face, 1)
            if random.random() > 0.5:
                face8 = cv2.flip(face8, 1)
            if random.random() > 0.5:
                face12 = cv2.flip(face12, 1)
            if random.random() > 0.5:
                face16 = cv2.flip(face16, 1)


        face_hr = face

        face_down8 = Image.fromarray(face8)
        face_down8 = face_down8.resize((8, 8))
        face_down8 = face_down8.resize((128, 128))
        face_down8 = np.asarray(face_down8)

        face_down12 = Image.fromarray(face12)
        face_down12 = face_down12.resize((12, 12))
        face_down12 = face_down12.resize((128, 128))
        face_down12 = np.asarray(face_down12)

        face_down16 = Image.fromarray(face16)
        face_down16 = face_down16.resize((16, 16))
        face_down16 = face_down16.resize((128, 128))
        face_down16 = np.asarray(face_down16)

        size = random.randint(8, 128)
        face = Image.fromarray(face)
        face = face.resize((size, size))
        face = face.resize((128, 128))
        face = np.asarray(face)

        data_dict = {
            'face': face_ToTensor(face),
            'face_hr': face_ToTensor(face_hr),
            'face_8': face_ToTensor(face_down8),
            'face_12': face_ToTensor(face_down12),
            'face_16': face_ToTensor(face_down16),
            'target': torch.LongTensor(self.targets[index]),
            'index': torch.LongTensor(np.float32([index]))
        }
        return data_dict

    def __len__(self):
        return len(self.subject_total)

    def get_targets(self):
        class_id = [path.split('/')[-2] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = len(unique_class_id)
        return targets.astype(np.int32), num_class

class SurvFaceDataset(torch.utils.data.Dataset):
    def __init__(self, is_aug):
        super(SurvFaceDataset, self).__init__()
        # mean_h, mean_w : 24.325, 19.92
        self.faces_path = glob(_survface_root + 'training_set/*/*')
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug

    def __getitem__(self, index):
        face = cv2.imread(self.faces_path[index])
        face = cv2.resize(face, (96//4, 112//4), cv2.INTER_CUBIC)
        if self.is_aug:
            if random.random() > 0.5:
                face = cv2.flip(face, 1)
        return face_ToTensor(face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return len(self.faces_path)

    def get_targets(self):
        class_id = [path.split(_survface_root + 'training_set/')[1].split('/')[0]
                    for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.int32), num_class


class SurvFace_verification(torch.utils.data.Dataset):
    def __init__(self):
        super(SurvFace_verification, self).__init__()
        self.root = _survface_root + 'Face_Verification_Test_Set/'
        self.image_root = self.root + 'verification_images/'
        pos = sio.loadmat(self.root + 'positive_pairs_names.mat')['positive_pairs_names']
        neg = sio.loadmat(self.root + 'negative_pairs_names.mat')['negative_pairs_names']
        self.image_files = np.vstack((pos, neg))
        self.labels = np.vstack((np.ones(len(pos)).reshape(-1, 1),
                                 np.zeros(len(neg)).reshape(-1, 1)))

    def __getitem__(self, index):
        sameflag = self.labels[index]
        img1 = cv2.imread(self.image_root + self.image_files[index][0][0])
        img2 = cv2.imread(self.image_root + self.image_files[index][1][0])

        # BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Resize
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img1 = img1.resize((128, 128), Image.BICUBIC)
        img2 = img2.resize((128, 128), Image.BICUBIC)

        # Mirror face trick
        img1_flip = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2_flip = img2.transpose(Image.FLIP_LEFT_RIGHT)

        img1 = np.asarray(img1)
        img2 = np.asarray(img2)
        img1_flip = np.asarray(img1_flip)
        img2_flip = np.asarray(img2_flip)

        return face_ToTensor(img1), face_ToTensor(img2), \
               face_ToTensor(img1_flip), face_ToTensor(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.labels)


class SCface_mugshot(torch.utils.data.Dataset):
    def __init__(self):
        super(SCface_mugshot, self).__init__()
        df = pd.read_csv(_scface_mugshot_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        assert len(self.faces_path) == 130, "Wrong number of SCface mugshot"
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, _= self.get_targets()

    def __getitem__(self, index):
        face = cv2.imread(_scface_root + self.faces_path[index])
        face = alignment2(face, self.landmarks[index].reshape(-1,2), scale=1)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = face.resize((128, 128), Image.BICUBIC)
        face_flip = face.transpose(Image.FLIP_LEFT_RIGHT)

        face = np.asarray(face)
        face_flip = np.asarray(face)

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
        self.targets, _= self.get_targets()

    def __getitem__(self, index):
        face = cv2.imread(_scface_root + self.faces_path[index])
        # face = crop_align_resize(face, self.landmarks[index].reshape(-1, 2), scale=1.3)
        face = alignment2(face, self.landmarks[index].reshape(-1,2), scale=1.)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = Image.fromarray(face)
        face = face.resize((128, 128), Image.BICUBIC)
        face_flip = face.transpose(Image.FLIP_LEFT_RIGHT)

        face = np.asarray(face)
        face_flip = np.asarray(face_flip)

        # face = cv2.resize(face, (128, 128), None, interpolation=cv2.INTER_CUBIC)
        # output = 'output/' + self.faces_path[index].split('/')[0] + '/'
        # if not os.path.exists(output):
        #     os.makedirs(output)
        # cv2.imwrite('output/' + self.faces_path[index], face)

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


class get_loader():
    def __init__(self, name, batch_size, is_aug=True, shuffle=True, drop_last=True, img_size=None, trialIndex=None, workers=8):
        if name == 'vggface2':
            dataset = VGGFace2Dataset(is_aug=is_aug)
            num_class = dataset.num_class
        elif name == 'survface':
            dataset = SurvFaceDataset(is_aug=is_aug)
            num_class = dataset.num_class
        elif name == 'survface_verification':
            dataset = SurvFace_verification()
            num_class = None
            shuffle = False
            drop_last = False
        elif name == 'scface_mugshots':
            dataset = SCface_mugshot()
            num_class = None
            shuffle = False
            drop_last = False
        elif 'scface_distance' in name:
            dataset = SCface_camera(name)
            num_class = None
            shuffle = False
            drop_last = False

        self.dataloader = DataLoader(dataset=dataset, num_workers=workers, batch_size=batch_size,
                                     pin_memory=False, shuffle=shuffle, drop_last=drop_last)
        self.num_class = num_class
        self.N = dataset.__len__()
        self.train_iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self.train_iter)
        except:
            del self.train_iter
            self.train_iter = iter(self.dataloader)
            data = next(self.train_iter)
        return data