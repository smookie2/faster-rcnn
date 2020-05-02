import skimage.io as io
from random import shuffle
import numpy as np


def genInputImg(ann_data, img_dir, classes, limit, img_shape):
	cat_ids = ann_data.getCatIds(catNms=classes)
	img_ids = ann_data.getImgIds(catIds=cat_ids)
	
	imgs = ann_data.loadImgs(img_ids)
	shuffle(imgs)

	for img in imgs[limit[0]:limit[1]]:

		# image data (h, w, channels)
		pix = io.imread('{}/{}'.format(img_dir, img['file_name']))

		# padding input img 
		x = np.zeros(img_shape, dtype='float32')
		if len(pix.shape) == 2:
			x[:pix.shape[0], :pix.shape[1], 0] = pix[:img_shape[0], :img_shape[1]]
			x[:pix.shape[0], :pix.shape[1], 1] = pix[:img_shape[0], :img_shape[1]]
			x[:pix.shape[0], :pix.shape[1], 2] = pix[:img_shape[0], :img_shape[1]]
		else:
			x[:pix.shape[0], :pix.shape[1], :] = pix[:img_shape[0], :img_shape[1], :]

		yield x, img['id']

def getBbox(ann_data, img_id, classes):
	cat_ids = ann_data.getCatIds(catNms=classes)
	ann_ids = ann_data.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=0)
	anns = ann_data.loadAnns(ids=ann_ids)

	return np.array([[ann['bbox'][1], ann['bbox'][0], 
				ann['bbox'][1]+ann['bbox'][3], ann['bbox'][0]+ann['bbox'][2], 
				ann['category_id']] for ann in anns])