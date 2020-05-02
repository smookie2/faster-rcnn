from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from datagen import genInputImg, getBbox
from utils import box2frame


img_dir = 'dataset/images'
ann_path = 'dataset/annotations/instances_face.json'
classes = ['head', None]
img_shape = [272, 480, 3]
ann_data = COCO(ann_path)

genX = genInputImg(ann_data=ann_data,
	img_dir=img_dir, classes=classes, 
	limit=[0, 100], img_shape=img_shape)

for i in range(100):
	x, img_id = next(genX)

	fig, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(x/255)

	bbox2d = getBbox(ann_data, img_id, classes)
	for bbox in bbox2d:
		frame = box2frame(box=bbox, apoint=[0, 0])
		ax.add_patch(Rectangle(
			(frame[0], frame[1]), frame[2], frame[3], 
			linewidth=1, 
			edgecolor='yellow',
			facecolor='none', 
			linestyle='-'))
	plt.show()
