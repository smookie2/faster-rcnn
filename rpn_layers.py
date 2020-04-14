from keras.layers import Conv2D

# To generate region proposals, 
# we slide a small network over the convolutional feature map output 
# by the last shared convolutional layer. 
# This small network takes as input an n × n spatial window of the input convolutional feature map.
# Each sliding window is mapped to a lower-dimensional feature (256-d for ZF and 512-d for VGG,
# with ReLU [33] following). This feature is fed into two sibling fully connected layers
# a box-regression layer (reg) and a box-classification layer (cls). 
# We use n = 3 in this paper, noting that the effective receptive field
# on the input image is large (171 and 228 pixels for ZF and VGG, respectively).
# This mini-network is illustrated at a single position in Figure 3 (left).
# Note that because the mini-network operates in a sliding-window fashion,
# the fully-connected layers are shared across all spatial locations.
# This architecture is naturally implemented with an n×n convolutional layer 
# followed by two sibling 1 × 1 convolutional layers (for reg and cls, respectively).
def rpn_layers(common_layers, k):
	'''
	Arguments:
		common_layers: shareable convolution layers (aka base block)
		k: number of anchors per point
	Return
		reg_layer: regression layer
		cls_layer: classification layer
	'''

	last_common_layer_features = common_layers.shape[3]

	intermediate_layer = Conv2D(filters=last_common_layer_features,
		kernel_size=(3,3),
		padding='same', 
		activation='relu', # max(0,x)
		name='RPN_INT', 
		data_format='channels_last')(common_layers)

	cls_layer = Conv2D(
		filters=2*k, 
		kernel_size=(1,1),
		padding='same', 
		activation='sigmoid', # (0,1)
		name='RPN_CLS', 
		data_format='channels_last')(intermediate_layer)

	reg_layer = Conv2D(
		filters=4*k, 
		kernel_size=(1,1),
		padding='same',
		activation='linear',
		name='RPN_REG', 
		data_format='channels_last')(intermediate_layer)

	return [cls_layer, reg_layer]
