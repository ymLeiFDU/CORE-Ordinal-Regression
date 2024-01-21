
import yaml
import torch
import argparse
import configs
import datetime
import sys
import os
import pickle

from train import train
from torch.utils.data import DataLoader
from data.dataset import MORPH
from models.vgg import vgg16_bn
from models.ema import EMA
from utils.losses import SORD_Loss, CELoss, MeanVarianceLoss
from utils.probordiloss import ProbOrdiLoss
from evaluate import evaluate
from torch.autograd import Variable
from utils.utils import *
from torchvision import models
from torch.optim.lr_scheduler import MultiStepLR, StepLR



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type = int, default = 10, help = 'num of epochs')
	parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
	parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
	parser.add_argument('--weight_decay', type = float, default = 0.0001, help = 'weight decay')
	parser.add_argument('--train_batch_size', type = int, default = 8, help = 'train batch size')
	parser.add_argument('--test_batch_size', type = int, default = 8, help = 'test batch size')
	parser.add_argument('--step_size', type = int, default = 10, help = 'lr decay step')
	parser.add_argument('--dataset', type = str, default = 'MORPH')
	parser.add_argument('--pretrained', type = int, default = 0)
	parser.add_argument('--model', type = str, default = 'fcn', help = 'save ckpt per ckpt_interval')
	parser.add_argument('--optimizer', type = str, default = 'SGD', help = 'choose optimizer')
	parser.add_argument('--fold', type = int, default = 1)
	parser.add_argument('--schedular_gamma', type = float, default = 0.2)
	parser.add_argument('--alpha', type = float, default = 1)
	
	parser.add_argument('--workers', type = int, default = 4, help = 'num of workers')
	parser.add_argument('--use_gpu', type = bool, default = True, help = 'whether or not using gpus')
	parser.add_argument('--gpus', type = str, default = 2, help = 'gpu numbers')
	parser.add_argument('--ckpt_interval', type = int, default = 10, help = 'save ckpt per ckpt_interval')
	parser.add_argument('--save_files', type = str, default = 'n', help = 'save running files or not')
	parser.add_argument('--run_tag', type = str, default = 'no tags', help = 'writting the tags here, blanks are invalid')
	
	
	opt = parser.parse_args()
	print('< Settings > : NCE+0.01MAE')
	print(opt)
	print('Training torch version: ', torch.__version__)
	with open('configs/paths.yaml', 'r') as f:
		cfgs = yaml.load(f)

	# ----------
	# Save files
	# ----------
	project_dir = '../Ordinal-Regression/PosetRegularizaiton'
	if not os.path.exists(project_dir + '/ckpt/' + opt.model):
		os.mkdir(project_dir + '/ckpt/' + opt.model)
	t = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	checkpoint_dir = project_dir + '/ckpt/' + opt.model + '/' + t + '-' + opt.run_tag
	print('Files saving dir: ', checkpoint_dir)
	args = cfgs['project_files']

	if opt.save_files == 'y':
		save_args(checkpoint_dir, args)
		logger = Tee(checkpoint_dir + '/log.txt', 'a')
	summary = TensorboardSummary(checkpoint_dir)
	writer = summary.create_summary()

	# ---------
	# Load data
	# ---------
	print('> load data ...')
	if opt.dataset.lower() == 'morph':
		train_data = MORPH(MORPH_path = cfgs['MORPH_path'], ratio = 0.8, mode = 'train', fold = opt.fold)
		train_loader = DataLoader(
			train_data,
			batch_size = opt.train_batch_size,
			shuffle = True,
			num_workers = opt.workers,
			pin_memory = True
			)


	train_num = train_data.__len__()
	test_num = test_data.__len__()

	# ----------------
	# Running settings
	# ----------------
	cuda = torch.cuda.is_available() and opt.use_gpu
	print('Saving files: ', opt.save_files)

	model = vgg16_bn(num_classes = num_classes)

	
	if cuda:
		if len(opt.gpus) > 1:
			gpus = [int(x) for x in opt.gpus.split(',')]
			torch.cuda.set_device(gpus[0])
			model = nn.DataParallel(model, device_ids = gpus)
			# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus)
			model = model.cuda()
		else:
			torch.cuda.set_device(int(opt.gpus))
			model = model.cuda()
	ema = EMA(model, alpha = 0.999)

	if opt.optimizer == 'Adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
	elif opt.optimizer == 'SGD':
		optimizer = torch.optim.SGD(
			model.parameters(), lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)

	scheduler = StepLR(optimizer, step_size = opt.step_size, gamma = opt.schedular_gamma)

	# loss functions
	loss_func = {
			'sord': SORD_Loss(ranks = np.arange(num_classes), loss_type = 'ce', metric = 'mae').cuda(),
			'ce': nn.CrossEntropyLoss().cuda(),
			'mv' : MeanVarianceLoss(lambda_1 = 0.2, lambda_2 = 0.05, start_age = 0, end_age = num_classes - 1).cuda(),
			}



	# -------
	#  train
	# -------
	save_time = datetime.datetime.now()

	best_cs = 0
	best_mae = 999
	best_cs_epoch = 0
	best_mae_epoch = 0
	best_model = model
	best_model_wts = model.state_dict()

	train_losses = []
	train_css = []
	train_maes = []

	test_losses = []
	test_css = []
	test_maes = []

	train_ce_losses = []
	train_kl_losses = []
	test_ce_losses = []
	test_kl_losses = []

	best_srcc = -1
	best_plcc = -1

	learned_lamb = []


	for epoch in range(opt.epochs):

		start_time = datetime.datetime.now()
		print('Epoch {} / {} ({}) Method: <{}>, {} | Bests: CS: {:.4f}(ep:{}), MAE: {:.4f}(ep:{}), SRCC: {:.4f}, PLCC: {:.4f}'.format(
			epoch, opt.epochs, opt.dataset, opt.run_tag, 
			t, best_cs, best_cs_epoch, best_mae, best_mae_epoch, best_srcc, best_plcc))
		model, train_loss, train_cs, train_mae, lamb, train_kl_loss, train_ce_loss, train_soft_loss = train(
			model = model,
			ema = ema,
			lr = opt.lr,
			epoch = epoch,
			train_num = train_num,
			max_epoch = opt.epochs,
			batch_size = opt.train_batch_size,
			optimizer = optimizer,
			scheduler = scheduler,
			loss_func = loss_func,
			lr_decay = 0.2,
			writer = writer,
			weight_decay = opt.weight_decay,
			train_loader = train_loader,
			lr_decay_epoch = opt.step_size,
			opt = opt)
		train_losses.append(train_loss)
		train_css.append(train_cs)
		train_maes.append(train_mae)
		learned_lamb.append(lamb)
		train_ce_losses.append(train_ce_loss)
		train_kl_losses.append(train_kl_loss)

		best_cs, best_mae, best_cs_epoch, best_mae_epoch, test_loss, test_cs, test_mae, best_model, best_model_wts, eval_ce_loss, eval_soft_loss = evaluate(
			epoch = epoch,
			opt = opt,
			checkpoint_dir = checkpoint_dir,
			test_loader = test_loader, 
			model = model,
			ema = ema,
			writer = writer,
			test_num = test_num,
			loss_func = loss_func,
			best_cs = best_cs,
			best_mae = best_mae,
			best_srcc = best_srcc,
			best_plcc = best_plcc,
			best_cs_epoch = best_cs_epoch,
			best_mae_epoch = best_mae_epoch,
			best_model = best_model,
			best_model_wts = best_model_wts,
			cuda = cuda)
		test_losses.append(test_loss)
		test_css.append(test_cs)
		test_maes.append(test_mae)
		test_ce_losses.append(eval_ce_loss)
		test_kl_losses.append(eval_kl_loss)

		end_time = datetime.datetime.now()
		print('elapsed time: {} s'.format((end_time - start_time).seconds))
		print()

		if epoch == opt.epochs - 1:
			if not os.path.exists(checkpoint_dir + '/trained_models'):
				os.mkdir(checkpoint_dir + '/trained_models')
			torch.save(best_model_wts, checkpoint_dir + '/trained_models' + '/MAE_{:.4f}_epoch_{}_weights.pth'.format(best_mae, best_mae_epoch))
			torch.save(best_model, checkpoint_dir + '/trained_models' + '/MAE_{:.4f}_epoch_{}_model.pth'.format(best_mae, best_mae_epoch))





















