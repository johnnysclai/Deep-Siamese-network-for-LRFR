import torch
import os, datetime, json, copy


def run(parser, dev):
	args = parser.parse_args()
	args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.name)
	## Set gpu ids
	str_ids = args.gpu_ids.split(',')
	args.gpu_ids = []
	for str_id in str_ids:
		gpu_id = int(str_id)
		if gpu_id >= 0:
			args.gpu_ids.append(gpu_id)
	if len(args.gpu_ids) > 0:
		torch.cuda.set_device(args.gpu_ids[0])
	## Set device
	args.device = 'cuda:{}'.format(args.gpu_ids[0]) if args.gpu_ids else 'cpu'
	if args.isTrain:
		## Set decay steps
		str_steps = args.decay_steps.split(',')
		args.decay_steps = []
		for str_step in str_steps:
			str_step = int(str_step)
			args.decay_steps.append(str_step)
		## Set names
		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		args.name = '{}_{}_{}_{}_{}'.format(current_time, args.dataset, args.backbone, args.head, args.loss)
		if args.use_dropout:
			args.name += '_dropout'
		if args.multi:
			args.name += '_multi'
		if args.lambda_dist > 0:
			args.name += f'_triplet_dist{args.lambda_dist}'
	## Print Options
	message = ''
	message += '----------------- Options ---------------\n'
	for k, v in sorted(vars(args).items()):
		comment = ''
		default = parser.get_default(k)
		if v != default:
			comment = '\t[default: %s]' % str(default)
		message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
	message += '----------------- End -------------------'
	print(message)
	if args.isTrain and not dev:
		args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.name)
		## Save Options as text
		os.makedirs(args.checkpoints_dir, exist_ok=True)
		file_name = os.path.join(args.checkpoints_dir, 'log.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')
		## Save options as json file
		file_name = os.path.join(args.checkpoints_dir, 'config.json')
		with open(file_name, 'w') as f:
			args_save = copy.deepcopy(args)
			args_save.isTrain = False
			json.dump(args_save.__dict__, f, indent=2)
	return args, message
