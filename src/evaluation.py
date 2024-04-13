import torch
import torch.nn as nn
from models.learner import CreateLearner
from benchmark import lfw_verification, scface_identification, survface_verification
from common.util import load_json2args

__checkpoint = '../checkpoints/paper/20200928-174452_vggface2_spherenet20_amsoftmax_celoss_multi_triplet_dist1.0/0100000_net_backbone.pth'
__config = '../checkpoints/paper/20200928-174452_vggface2_spherenet20_amsoftmax_celoss_multi_triplet_dist1.0/config.json'

if __name__ == '__main__':
    args = load_json2args(__config)
    learner = CreateLearner(args)
    backbone = learner.backbone
    backbone.load_state_dict(torch.load(__checkpoint))
    # if len(args.gpu_ids) > 1:
    #     backbone = nn.DataParallel(backbone)

    # Create evaluation objects
    lfw_8 = lfw_verification.LFW(args=args, img2_size=8)
    lfw_12 = lfw_verification.LFW(args=args, img2_size=12)
    lfw_16 = lfw_verification.LFW(args=args, img2_size=16)
    lfw_20 = lfw_verification.LFW(args=args, img2_size=20)
    lfw_128 = lfw_verification.LFW(args=args, img2_size=128)
    lfws = [lfw_8, lfw_12, lfw_16, lfw_20, lfw_128]
    for lfw in lfws:
        lfw.run(backbone)
    scface_identification.run(args, backbone,)
    survface_verification.run(args, backbone)