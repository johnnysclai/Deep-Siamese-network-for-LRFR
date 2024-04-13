import torch
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import pandas as pd
import argparse
from sklearn import preprocessing
import sys
from tqdm import tqdm
sys.path.insert(0, '../../src')
from common.util import alignment

parser = argparse.ArgumentParser(description='Loader')
parser.add_argument('--vggface2_root', default='../../../VGGFace2/', type=str)
parser.add_argument('--vggface2_landmarks', default='../../../VGGFace2/bb_landmark/loose_landmark_train_test_remove_lfw_megaface.csv', type=str)


class VGGFace2Dataset(torch.utils.data.Dataset):
    def __init__(self, args, size=False, is_aug=True):
        super(VGGFace2Dataset, self).__init__()
        # Original #subjects(#images): xxx(3141890)
        df = pd.read_csv(args.vggface2_landmarks, delimiter=",")
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:,0]
        self.landmarks = numpyMatrix[:,1:]
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug
        self.size = size
        self.args = args

    def __getitem__(self, index):
        path = self.args.vggface2_root + self.faces_path[index] + '.jpg'
        if 'VGGFace2/train/' in path:
            path_new = path.replace('train', 'train_test_128x128/')
        elif 'VGGFace2/test/' in path:
            path_new = path.replace('test', 'train_test_128x128/')
        folder = os.path.join('/', *path_new.split('/')[:-1])
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except:
                pass
        face = cv2.imread(path)
        if face is None:
            print(path)
        face = alignment(face, self.landmarks[index].reshape(-1,2), size=128)
        cv2.imwrite(path_new, face)
        return torch.LongTensor(self.targets[index])

    def __len__(self):
        return self.faces_path.shape[0]

    def get_targets(self):
        class_id = [path.split('/')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.float32), num_class


def get_loader(batch_size):
    args = parser.parse_args()
    dataset = VGGFace2Dataset(args, is_aug=False)
    total_images = dataset.__len__()
    loader = DataLoader(dataset=dataset, num_workers=12, batch_size=batch_size, pin_memory=True, shuffle=False)
    return loader, total_images


if __name__ == '__main__':
    loader, total_images = get_loader(1024)
    counter = 0
    pbar = tqdm(loader, ncols=0)
    for target in pbar:
        counter += target.shape[0]
        pbar.set_description(desc='{}/{}'.format(counter, total_images))