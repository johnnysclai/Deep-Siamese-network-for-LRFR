import torch

import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from common.loader import get_loader


def run(args, net, step=None):
    net.eval()
    dataloader = get_loader('survface_verification', 128, workers=2).dataloader
    scores, labels = None, None
    with torch.no_grad():
        for index, (img1, img2, img1_flip, img2_flip, targets) in enumerate(dataloader):
            img1, img2 = img1.to(args.device), img2.to(args.device)
            img1_flip, img2_flip = img1_flip.to(args.device), img2_flip.to(args.device)
            features11, features12 = net(img1), net(img1_flip)
            features21, features22 = net(img2), net(img2_flip)
            # features1 = torch.cat((features11, features12), dim=1)
            # features2 = torch.cat((features21, features22), dim=1)
            cosineSimilarity = torch.nn.CosineSimilarity()(features11, features21)
            if index == 0:
                scores = cosineSimilarity.detach().cpu().numpy().reshape((-1,1))
                labels = targets.numpy().reshape((-1,1))
            else:
                scores = np.vstack((scores, cosineSimilarity.detach().cpu().numpy().reshape((-1,1))))
                labels = np.vstack((labels, targets.numpy().reshape((-1,1))))
    # scores = scores.reshape(-1)
    # labels = labels.reshape(-1)
    # labels = labels[np.argsort(scores)]
    # scores = scores[np.argsort(scores)]
    # labels = labels == 1
    #
    # for threshold in scores:
    #     predict_label = scores > threshold
    #     print('asd')

    fpr, tpr, _ = roc_curve(labels, scores)

    tpr0 = None
    tpr1 = None
    tpr2 = None
    tpr3 = None
    for f, t in zip(fpr, tpr):
        if f > 0.001 and tpr0 is None:
            if t == last_tpr:
                tpr0 = t
            else:
                tpr0 = (f - last_fpr)/(t-last_tpr) * (0.001-f) + t
        if f > 0.01 and tpr1 is None:
            if t == last_tpr:
                tpr1 = t
            else:
                tpr1 = (f - last_fpr)/(t-last_tpr) * (0.01-f) + t
        if f > 0.1 and tpr2 is None:
            if t == last_tpr:
                tpr2 = t
            else:
                tpr2 = (f - last_fpr)/(t-last_tpr) * (0.1-f) + t
        if f > 0.3 and tpr3 is None:
            if t == last_tpr:
                tpr3 = t
            else:
                tpr3 = (f - last_fpr)/(t-last_tpr) * (0.3-f) + t

        last_fpr = f
        last_tpr = t
    roc_auc = auc(fpr, tpr)
    if step is not None:
        message = 'SurvFace verification auc={:.4f} at {}iter'.format(roc_auc, step)
    else:
        message = 'SurvFace verification auc={:.4f} at testing'.format(roc_auc, step)

    message += '{:4f}(30%) {:4f}(10%) {:4f}(1%) {:4f}(0.1%)'.format(tpr3, tpr2, tpr1, tpr0)
    print(message)

    if step is not None:
        log_name = os.path.join(args.checkpoints_dir, 'log.txt')
        with open(log_name, "a") as log_file:
            log_file.write('\n' + message)