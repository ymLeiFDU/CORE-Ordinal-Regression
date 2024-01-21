import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from data.dataset import *
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler

from models.utils import adjacent_matrix
from utils.utils import Metrics
from utils.losses import CELoss, MAELoss, RCE, NCE, Entropy, NFLoss, SORD_Loss

from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
	
def train(
		model,
		ema,
		lr,
		epoch,
		max_epoch,
		batch_size,
		train_num,
		optimizer,
		loss_func,
		scheduler,
		lr_decay,
		writer,
		weight_decay,
		train_loader,
		lr_decay_epoch,
		opt,
		):
	

	model.train()
	evaluator = Metrics()

	running_corrects = 0

	MAE = 0
	EXP_MAE = 0

	css = []
	losses = []
	ce_losses = []
	kl_losses = []
	soft_losses = []
	poset_losses = []

	spearman = SpearmanCorrCoef()
	pearson = PearsonCorrCoef()
	srcc_values = []
	plcc_values = []

	scheduler.step()

	tbar = tqdm(train_loader)
	acc_datasets = ['hid', 'busi', 'dr', 'adience', 'lung', 'ava', 'aadb']
	for batch_i, (input, label, mh_targets) in enumerate(tbar):
		input_var = input.type(torch.cuda.FloatTensor).squeeze(1)
		label = label.type(torch.cuda.LongTensor)

		label_graph = adjacent_matrix(label)

		optimizer.zero_grad()


		output, emb_matrix, emb, dualkl_loss, lamb = model(input_var, label_graph, label)
		

		ce_loss = loss_func['ce'](output, label)
		soft_loss = loss_func['sord'](output, label)
		m_loss, v_loss = loss_func['mv'](output, label)
		loss = ce_loss + opt.alpha * dualkl_loss



		loss.backward()
		optimizer.step()
		# ema.update_params()

		losses.append(loss.data.cpu().numpy())
		ce_losses.append(ce_loss.data.cpu().numpy())
		soft_losses.append(soft_loss.data.cpu().numpy())

		mae, exp_mae, acc = evaluator.cal_mae_acc_cls(output, label)
		if opt.dataset.lower() in acc_datasets:
			_, pred_label = torch.max(output.data, 1)
			if opt.dataset.lower() == 'ava' or opt.dataset.lower() == 'aadb':
				srcc = spearman(pred_label.type(torch.FloatTensor), label.type(torch.FloatTensor))
				plcc = pearson(pred_label.type(torch.FloatTensor), label.type(torch.FloatTensor))
				srcc_values.append(srcc)
				plcc_values.append(plcc)
			running_corrects += torch.sum(pred_label == label.data)
		else:
			cs, correct = evaluator.CS_score(output, label, level = 5)
			running_corrects += correct
		

		MAE += mae
		EXP_MAE += exp_mae

		tbar.set_description('Train loss: %.5f' % (loss / (batch_i + 1)))


	# -------------------
	#  calculate metrics
	# -------------------
	mean_mae = MAE / train_num
	mean_loss = np.mean(losses)
	mean_ce_loss, mean_kl_loss, mean_soft_loss = np.mean(ce_losses), np.mean(kl_losses), np.mean(soft_losses)
	mean_srcc = np.mean(srcc_values)
	mean_plcc = np.mean(plcc_values)
	CS = running_corrects / np.float(train_num)


	print('<Train> [Loss]: {:.8f} |[MAE]: {:.4f}, [CS]: {:.4f}, [SRCC]: {:.4f}, [PLCC]: {:.4f} | <Train>'.format(
		mean_loss, mean_mae, CS, mean_srcc, mean_plcc))

	# ema.update_buffer()

	return model, mean_loss, CS, mean_mae, lamb.data.cpu().numpy(), mean_kl_loss, mean_ce_loss, mean_soft_loss


























