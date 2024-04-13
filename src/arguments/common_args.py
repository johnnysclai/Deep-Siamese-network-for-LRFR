import argparse
from common.util import str2bool


def get_args():
	parser = argparse.ArgumentParser(description='Deep face recognition')
	## Model settings
	parser.add_argument('--backbone', default='spherenet20', type=str,
	                    help='spherenetx/seresnet50/densenet121/densenet161')
	parser.add_argument('--use_pool', default=False, type=str2bool, help='use global pooling?')
	parser.add_argument('--use_dropout', default=False, type=str2bool, help='use dropout?')
	parser.add_argument('--feature_dim', default=512, type=int, help='feature dimension')
	parser.add_argument('--gpu_ids', default='0,1', type=str, help='gpu ids: e.g. 0 0,1,2, 0,2.')
	return parser
