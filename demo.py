from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


from common_layers import common_layers_5 as common_layers
from rpn_layers import rpn_layers

def rpn_cls_loss_with_anchors(num_of_anchors_per_point):
	def rpn_cls_loss(y_true, y_pred):
		# y_true (batch_size, w, h, 2k+2k)
		# y_pred (batch_size, w, h, 2k)
		return K.mean(y_true[:, :, :, 2*num_of_anchors_per_point:] * K.binary_crossentropy(y_true[:, :, :, :2*num_of_anchors_per_point], y_pred[:, :, :, :]))
	return rpn_cls_loss

def rpn_reg_loss_with_anchors(num_of_anchors_per_point):
	def rpn_reg_loss(y_true, y_pred):
		# y_true (batch_size, w, h, 2*4k)
		# y_pred (batch_size, w, h, 4k)
		# if anchor is so large, this effects converging of CLS, DENGEROUS, let's consider to normalize compute_bbe
		return K.mean(y_true[:, :, :, 4*num_of_anchors_per_point:] * K.abs(y_true[:, :, :, :4*num_of_anchors_per_point] - y_pred[:, :, :, :]))
	return rpn_reg_loss

img_shape = (64, 64, 3)
anchor_point_num = 3 # k

input_layer = Input(shape=img_shape)
common_layers = common_layers(input_layer)
rpn_layers = rpn_layers(common_layers, anchor_point_num)

rpn_model = Model(inputs=input_layer, outputs=rpn_layers, name="RPN")

rpn_model.compile(optimizer=Adam(lr=0.0001),
	loss=[rpn_cls_loss_with_anchors(anchor_point_num), rpn_reg_loss_with_anchors(anchor_point_num)])
rpn_model.summary()