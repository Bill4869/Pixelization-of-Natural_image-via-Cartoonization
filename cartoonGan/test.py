import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from cartoonGan.network.Transformer import Transformer

def cartoonize(opt):
	
	valid_ext = ['.jpg', '.png']

	if not os.path.exists(opt.cartoon_output): os.mkdir(opt.cartoon_output)

	# load pretrained model
	model = Transformer()
	model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))
	model.eval()

	if opt.gpu > -1:
		print('GPU mode')
		model.cuda()
	else:
		print('CPU mode')
		model.float()

	for files in os.listdir(opt.natural_input):
		ext = os.path.splitext(files)[1]
		if ext not in valid_ext:
			continue
		# load image
		input_image = Image.open(os.path.join(opt.natural_input, files)).convert("RGB")
		
		# resize image, keep aspect ratio
		h = input_image.size[0]
		w = input_image.size[1]
		ratio = h *1.0 / w
		if ratio > 1:
			h = opt.loadSize
			w = int(h*1.0/ratio)
		else:
			w = opt.loadSize
			h = int(w * ratio)
		input_image = input_image.resize((h, w), Image.BICUBIC)

		# input_image = input_image.resize((int(opt.loadSize), int(opt.loadSize)), Image.BICUBIC)

		input_image = np.asarray(input_image)
		# RGB -> BGR
		input_image = input_image[:, :, [2, 1, 0]]
		input_image = transforms.ToTensor()(input_image).unsqueeze(0)
		# preprocess, (-1, 1)
		input_image = -1 + 2 * input_image 
		if opt.gpu > -1:
			input_image = Variable(input_image, volatile=True).cuda()
		else:
			input_image = Variable(input_image, volatile=True).float()
		# forward
		output_image = model(input_image)
		output_image = output_image[0]
		# BGR -> RGB
		output_image = output_image[[2, 1, 0], :, :]
		# deprocess, (0, 1)
		output_image = output_image.data.cpu().float() * 0.5 + 0.5
		# save
		vutils.save_image(output_image, os.path.join(opt.cartoon_output, files[:-4] + '_' + opt.style + '.png'))



print('Done!')
