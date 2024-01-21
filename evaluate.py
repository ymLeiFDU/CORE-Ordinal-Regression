import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import *
from torch.autograd import Variable
from terminaltables import AsciiTable
from tqdm import tqdm
from utils.utils import output2label
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.utils import Metrics
from utils.losses import CELoss
from train import adjacent_matrix

from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef


def evaluate(epoch, 
	test_loader, 
	model, 
	ema,
	checkpoint_dir,
	opt,
	writer,
	test_num,
	loss_func,
	best_cs,
	best_mae,
	best_srcc,
	best_plcc,
	best_cs_epoch,
	best_mae_epoch,
	best_model,
	best_model_wts,
	cuda = True):

	epoch_loss = 0

	running_corrects = 0
	losses = []
	ce_losses = []
	ordinal_losses = []
	soft_losses = []
	poset_losses = []
	MAE = 0
	EXP_MAE = 0
	css = []

	spearman = SpearmanCorrCoef()
	pearson = PearsonCorrCoef()
	srcc_values = []
	plcc_values = []

	# ema.apply_shadow()
	# ema.model.eval()
	# ema.model.cuda()

	evaluator = Metrics()

	tbar = tqdm(test_loader)
	acc_datasets = ['hid', 'busi', 'dr', 'adience', 'lung', 'ava', 'aadb']
	for batch_i, (input, label, mh_targets) in enumerate(tbar):

		input_var = input.type(torch.cuda.FloatTensor).squeeze(1)
		label = label.type(torch.cuda.LongTensor)
		
		label_graph = adjacent_matrix(label)

		with torch.no_grad():
			output, emb_matrix, emb, dualkl_loss, lamb = model(input_var, label_graph, label)


		ce_loss = loss_func['ce'](output, label)
		soft_loss = loss_func['sord'](output, label)
		m_loss, v_loss = loss_func['mv'](output, label)
		loss = ce_loss 

		# compute MAE
		losses.append(loss.data.cpu().numpy())
		ce_losses.append(ce_loss.data.cpu().numpy())
		ordinal_losses.append(kl_loss.data.cpu().numpy())
		soft_losses.append(soft_loss.data.cpu().numpy())

		mae, exp_mae, acc = evaluator.cal_mae_acc_cls(output, label)
		if opt.dataset.lower() in acc_datasets:
			_, pred_label = torch.max(output.data, 1)
			running_corrects += torch.sum(pred_label == label.data)
			if opt.dataset.lower() == 'aadb':
				srcc = spearman(pred_label.type(torch.FloatTensor), label.type(torch.FloatTensor))
				plcc = pearson(pred_label.type(torch.FloatTensor), label.type(torch.FloatTensor))
				srcc_values.append(srcc)
				plcc_values.append(plcc)
		else:
			cs, correct = evaluator.CS_score(output, label, level = 5)
			running_corrects += correct

		MAE += mae
		EXP_MAE += exp_mae

		tbar.set_description('Test loss: %.5f' % (loss / (batch_i + 1)))

	# -------------------
	#  calculate metrics
	# -------------------
	mean_mae = MAE / test_num
	mean_exp_mae = EXP_MAE / test_num
	mean_loss = np.mean(losses)
	mean_ce_loss, mean_soft_loss = np.mean(ce_losses), np.mean(soft_losses)
	mean_srcc = np.mean(srcc_values)
	mean_plcc = np.mean(plcc_values)
	CS = running_corrects / np.float(test_num)


	if best_cs < CS:
		best_cs = CS
		best_cs_epoch = epoch

	if best_mae > mean_mae:
		best_mae = mean_mae
		best_mae_epoch = epoch
		best_model_wts = model.state_dict()
		best_model = model


	print('<Eval> [Loss]: {:.8f} | [MAE]: {:.4f}, [EXP_MAE]: {:.4f}, [CS]: {:.4f} | <Eval>'.format(
		mean_loss, mean_mae, mean_exp_mae, CS))

	# ema.restore()

	return best_cs, best_mae, best_cs_epoch, best_mae_epoch, mean_loss,\
	 CS, mean_mae, best_model, best_model_wts, \
	 mean_ce_loss, mean_soft_loss
















