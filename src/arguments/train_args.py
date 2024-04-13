from . import common_args, modify_args
from common.util import str2bool


def get_args(dev=False):
	parser = common_args.get_args()
	parser.add_argument('--isTrain', default=True, type=str2bool, help='is train?')
	## Training settings
	parser.add_argument('--dataset', default='vggface2', type=str,
	                    help='vggface2/survface')
	parser.add_argument('--head', default='amsoftmax', type=str,
	                    help='softmax/asoftmax/arcface/amsoftmax')
	parser.add_argument('--loss', default='celoss', type=str, help='celoss/focalloss')
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
	parser.add_argument('--bs', default=128, type=int, help='default: 256')
	parser.add_argument('--iterations', default=100000, type=int, help='50000 for vggface2')
	parser.add_argument('--decay_steps', default='40000, 60000, 80000', type=str,
	                    help='40000, 60000, 80000 for vggface2')
	parser.add_argument('--s', default=30, type=float,
	                    help='amsoftmax(30), cosface/arcface(64)')
	parser.add_argument('--m_3', default=0.35, type=float, help='margin of AMSoftmax/CosineFace')
	parser.add_argument('--gamma', default=2, type=float, help='gamma of Focal loss')
	## Other setting
	parser.add_argument('--eval_freq', default=1000, type=int, help='frequency of evaluation')
	parser.add_argument('--name', type=str, default='master', help='name of the experiment.')
	parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
	## Testing parameter
	parser.add_argument('--lambda_dist', default=1., type=float, help='lambda for dist loss')
	parser.add_argument('--multi', default=True, type=str2bool, help='use multi CE?')
	args, message = modify_args.run(parser, dev=dev)
	return args, message
