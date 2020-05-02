def box2frame(box, apoint=[0.5, 0.5]):
	'''
	Convert [y1, x1, y2, x2] to [x, y, w, h]
	'''

	return [
		(box[1] + apoint[1]*(box[3]-box[1])),
		(box[0] + apoint[0]*(box[2]-box[0])),
		(box[3] - box[1]),
		(box[2] - box[0])
	]