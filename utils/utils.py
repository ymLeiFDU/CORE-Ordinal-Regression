
from __future__ import division

import torch
import torch.nn as nn
import json
import os
import sys
import os
import numpy as np
import cv2
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from shutil import copyfile
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


def weights_init_normal(m):
	# ------------------
	# initialize weights
	# ------------------
	classname = m.__class__.__name__
	if classname.find("Conv2d") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


def convert_vgg_weights(pretrained_vgg_dict, target_net = None):

	if target_net is None:
		raise Exception("Param 'target_net' should be given ", target_net)
	else:
		mynet_dict = target_net.state_dict()

		pretrained_dict = list(pretrained_vgg_dict.items())

		j = 0
		for key, value in mynet_dict.items():
			layer_name, weights = pretrained_dict[j]
			mynet_dict[key] = weights
			j += 1

	return mynet_dict


def load_pre_weights(target_net, copy_net = 'vgg16'):

	model_dict = target_net.state_dict()

	pretrained_dict = model_zoo.load_url(vgg.model_urls[copy_net])
	pretrained_dict = convert_vgg_weights(pretrained_vgg_dict = pretrained_dict, target_net = target_net)
	
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	target_net.load_state_dict(model_dict)

	return target_net


def save_args(ckpt_dir, args):
	# -----------------------------------
	# save running files (.py, .sh, etc.)
	# -----------------------------------
	if not os.path.exists('ckpt_dir'):
		os.mkdir(ckpt_dir)

	os.mkdir(ckpt_dir + '/models')
	os.mkdir(ckpt_dir + '/utils')
	os.mkdir(ckpt_dir + '/data')
	os.mkdir(ckpt_dir + '/results')

	print('Saving running files ...')
	for f in args:
		print(f)
		copyfile(f, ckpt_dir + '/' + f)


class Tee(object):
	def __init__(self, name, mode):
		self.file = open(name, mode)
		self.stdout = sys.stdout
		sys.stdout = self

	def __def__(self):
		sys.stdout = self.stdout
		self.file.close()

	def write(self, data):
		if not '...' in data:
			self.file.write(data)

		self.stdout.write(data)
	
	def flush(self):
		self.file.flush()


def output2label(out_list):

	cvt_label = torch.LongTensor((len(out_list)))
	for i, j in enumerate(out_list):
		value, pred = torch.max(j, 1)
		cvt_label[i] = pred
	cvt_label = cvt_label.unsqueeze(0)
	return cvt_label.cuda()


class Metrics(object):

	def __init__(self):
		super(Metrics, self).__init__()
		
	def CS_score(self, pred_label, label, level = 5):

		bs = pred_label.size(0)

		pred_label = F.softmax(pred_label, -1).data.cpu().numpy()
		pred_label = np.argmax(pred_label, axis = 1)

		label = label.data.cpu().numpy()
		label = np.array(label)

		abs_array = np.abs(pred_label - label)
		abs_bool = abs_array <= level
		abs_bin = abs_bool.astype(np.uint8)
		correct = np.sum(abs_bin)

		CS = correct / bs
		return CS, correct

	def cal_mae_acc_cls(self, logits, targets):
		s_dim, out_dim = logits.shape
		probs = F.softmax(logits, -1)
		probs_data = probs.cpu().data.numpy()
		target_data = targets.cpu().data.numpy()
		max_data = np.argmax(probs_data, axis=1)
		label_arr = np.array(range(out_dim))
		exp_data = np.sum(probs_data * label_arr, axis=1)

		mae = sum(abs(max_data - target_data)) * 1.0
		exp_mae = sum(abs(exp_data - target_data)) * 1.0
		acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

		return mae, exp_mae, acc

	def cal_mae_acc_reg(self, logits, targets, is_sto=True):

		logits = torch.argmax(logits, dim = 1)

		assert logits.view(-1).shape == targets.shape, "logits {}, targets {}".format(
			logits.shape, targets.shape)


		logits = logits.cpu().data.numpy().reshape(-1)
		targets = targets.cpu().data.numpy()
		mae = sum(abs(logits - targets)) * 1.0
		acc = sum(np.rint(logits) == targets) * 1.0 / len(targets)

		return mae, acc

	def cal_mae(self, logits, targets):
		print(logits.size(), targets.size())


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        show_num = 3
        grid_image = make_grid(image[:show_num].clone().cpu().data, show_num, normalize=True)
        writer.add_image('Image', grid_image, global_step)

























