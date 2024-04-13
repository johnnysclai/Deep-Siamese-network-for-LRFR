import os
import torch
import numpy as np
from loader import get_loader
from common.util import tensor_pair_cosine_distance_matrix


def run(net_hr, net_lr, args, step=None):
    net_hr.eval()
    net_lr.eval()
    dataloader = get_loader('scface_mugshots', 128).dataloader
    features_gallery1_total = torch.Tensor(np.zeros((130, args.feature_dim), dtype=np.float32)).to(args.device)
    features_gallery2_total = torch.zeros_like(features_gallery1_total)
    labels_gallery = torch.Tensor(np.zeros((130, 1), dtype=np.float32)).to(args.device)
    with torch.no_grad():
        bs_total = 0
        for index, (img1, img1_flip, targets) in enumerate(dataloader):
            bs = len(targets)
            img1, img1_flip = img1.to(args.device), img1_flip.to(args.device)
            features1, features2 = net_hr(img1), net_hr(img1_flip)
            features_gallery1_total[bs_total:bs_total + bs] = features1
            features_gallery2_total[bs_total:bs_total + bs] = features2
            labels_gallery[bs_total:bs_total + bs] = targets
            bs_total += bs
        assert bs_total == 130, '# of mugshot should be 130'

    QUERIES = ['distance1', 'distance2', 'distance3']
    for query in QUERIES:
        dataloader = get_loader('scface_{}'.format(query), 128).dataloader
        features_query1_total = torch.Tensor(np.zeros((650, args.feature_dim), dtype=np.float32)).to(args.device)
        features_query2_total = torch.zeros_like(features_query1_total)
        labels_query = torch.Tensor(np.zeros((650, 1), dtype=np.float32)).to(args.device)
        with torch.no_grad():
            bs_total = 0
            for index, (img1, img1_flip, targets) in enumerate(dataloader):
                bs = len(targets)
                img1, img1_flip = img1.to(args.device), img1_flip.to(args.device)
                features1, features2 = net_lr(img1), net_lr(img1_flip)
                features_query1_total[bs_total:bs_total + bs] = features1
                features_query2_total[bs_total:bs_total + bs] = features2
                labels_query[bs_total:bs_total + bs] = targets
                bs_total += bs
            assert bs_total == 650, '# of {} images should be 650'.format(query)

        features_gallery1_total -= features_gallery1_total.mean(dim=0)
        features_gallery2_total -= features_gallery2_total.mean(dim=0)
        features_query1_total -= features_query1_total.mean(dim=0)
        features_query2_total -= features_query2_total.mean(dim=0)
        ## Matching
        for cal_type in ['concat', 'normal']:  # cal_type: concat/sum/normal
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
                log_name = os.path.join(args.checkpoints_dir, args.name, 'log.txt')
                with open(log_name, "a") as log_file:
                    log_file.write('\n' + message)
