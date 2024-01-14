from __future__ import division, print_function

import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torch.nn import CTCLoss
from torchvision.transforms import Compose

import datasets.dataset as dataset
import utils
from config import *
from datasets.ae_transforms import *
from datasets.imprint_dataset import Rescale as IRescale
from models.model import ModelBuilder


class BaseHTR(object):
	def __init__(self, opt, dataset_name='iam', reset_log=False):
		self.opt = opt
		self.mode = self.opt.mode
		self.dataset_name = dataset_name
		self.stn_nc = self.opt.stn_nc
		self.cnn_nc = self.opt.cnn_nc
		self.criterion = CTCLoss(blank=0, reduction='sum', zero_infinity=True)
		self.label_transform = None
		self.test_transforms = Compose([
			IRescale(max_width=self.opt.imgW, height=self.opt.imgH),
			ToTensor()
		])
		self.val1_iter = self.opt.val1_iter # Number of train data batches that will be validated
		self.val2_iter = self.opt.val2_iter # Number of validation data batches that will be validated
		# self.val_metric = 'cer'
		self.identity_matrix = torch.tensor([1, 0, 0, 0, 1, 0],
									   dtype=torch.float).cuda()
		
		##################################################################
		self.test_root = self.opt.valRoot

		random.seed(self.opt.manualSeed)
		np.random.seed(self.opt.manualSeed)
		torch.manual_seed(self.opt.manualSeed)

		cudnn.deterministic = True
		cudnn.benchmark = False
		cudnn.enabled = True

		if torch.cuda.is_available() and not self.opt.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			self.opt.gpu_id = list(map(int, self.opt.gpu_id.split(',')))
			torch.cuda.set_device(self.opt.gpu_id[0])

		self.test_data = dataset.lmdbDataset(
			root=self.test_root,
			voc=self.opt.alphabet,
			transform=self.test_transforms,
			label_transform=self.label_transform,
			voc_type='file',
			return_list=True
		)
		self.converter = utils.strLabelConverter(
			self.test_data.id2char,
			self.test_data.char2id,
			self.test_data.ctc_blank
		)
		self.nclass = self.test_data.rec_num_classes

		crnn = ModelBuilder(
			96, 256,
			[48,128], [96,256],
			20, [0.05, 0.05],
			'none',
			256, 1, 1,
			self.nclass,
			STN_type='TPS',
			nheads=1,
			stn_attn=None,
			use_loc_bn=False,
			loc_block = 'LocNet',
			CNN='ResCRNN'
		)
		if self.opt.cuda:
			crnn.cuda()
			crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id, dim=1)
		else:
			crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id)
		print('Using pretrained model', self.opt.pretrained)
		crnn.load_state_dict(torch.load(self.opt.pretrained))
		self.model = crnn

		self.init_variables()
		print('Classes: ', self.test_data.voc)
		print('#Test Samples: ', self.test_data.nSamples)

		data_loader = torch.utils.data.DataLoader(
			self.test_data,
			batch_size=64,
			num_workers=2,
			pin_memory=True,
			collate_fn=dataset.collatedict(),
			drop_last=False
		)
		self.model.eval()
		gts = []
		decoded_preds = []
		val_iter = iter(data_loader)
		max_iter = min(np.inf, len(data_loader))
		with torch.no_grad():
			for i in range(max_iter):
				cpu_images, cpu_texts = next(val_iter)
				utils.loadData(self.image, cpu_images)
				output_dict = self.model(self.image)
				batch_size = cpu_images.size(0)

				preds = F.log_softmax(output_dict['probs'], 2)

				preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
				_, preds = preds.max(2)
				preds = preds.transpose(1, 0).contiguous().view(-1)
				decoded_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

				gts += list(cpu_texts)
				decoded_preds += list(decoded_pred)

		directory = self.opt.out
		dataset_name = self.opt.mode
		writepath1 = directory + '/' + dataset_name + "_gt_and_predicted_text" + ".txt" 
		print(writepath1)
		mode = 'a' if os.path.exists(writepath1) else 'w'
		with open(writepath1, mode) as f1:
			for target, pred in zip(gts, decoded_preds):         
				print(target, pred)
				f1.write(str(target))
				f1.write("\t")
				f1.write(str(pred))
				f1.write("\n") 
		return


	def init_variables(self):
		self.image = torch.FloatTensor(self.opt.batchSize, 3, self.opt.imgH, self.opt.imgH)
		self.text = torch.LongTensor(self.opt.batchSize * 5)
		self.length = torch.LongTensor(self.opt.batchSize)
		if self.opt.cuda:
			self.image = self.image.cuda()
			self.criterion = self.criterion.cuda()
			self.text = self.text.cuda()
			self.length = self.length.cuda()
		self.image = Variable(self.image)
		self.text = Variable(self.text)
		self.length = Variable(self.length)


if __name__ == "__main__":
	opt = parser.parse_args()
	obj = BaseHTR(opt)
	obj.run()
