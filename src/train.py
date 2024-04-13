from models.learner import CreateLearner
from benchmark import lfw_verification, scface_identification, survface_verification
from common.loader import get_loader
from common.util import save_log
from arguments import train_args
from tqdm import tqdm
import torch

if __name__ == '__main__':
	args, message = train_args.get_args()
	dataloader = get_loader(name=args.dataset, batch_size=args.bs, workers=6)
	learner = CreateLearner(args)
	learner.train_setup(class_num=dataloader.num_class)  # setup the optimizer and scheduler
	## Setup benchmarks
	lfw_8 = lfw_verification.LFW(args=args, img2_size=8)
	lfw_12 = lfw_verification.LFW(args=args, img2_size=12)
	lfw_16 = lfw_verification.LFW(args=args, img2_size=16)
	lfw_20 = lfw_verification.LFW(args=args, img2_size=20)
	lfw_128 = lfw_verification.LFW(args=args, img2_size=128)
	lfws = [lfw_8, lfw_12, lfw_16, lfw_20, lfw_128]
	pbar = tqdm(range(1, args.iterations + 1), ncols=0)
	for steps in pbar:
		learner.train_enable()
		learner.optimize_parameters(dataloader.next())  # forward and backprop
		learner.update_learning_rate()  # update learning rate
		losses = learner.get_current_losses()
		if steps % args.eval_freq == 0:
			learner.train_disable()
			print('')
			for lfw in lfws:
				lfw.run(learner.backbone, step=steps)
			scface_identification.run(args, learner.backbone, step=steps)
			survface_verification.run(args, learner.backbone, step=steps)
			save_log(losses, args)
		if steps % 100 == 0:
			torch.cuda.empty_cache()
		# display
		description = ""
		for name, value in losses.items():
			description += '{}: {:.4f} '.format(name, value)
		pbar.set_description(desc=description)
	learner.save_networks(steps)
