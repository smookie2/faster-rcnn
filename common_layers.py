from keras.layers import Conv2D, MaxPooling2D

# A Region Proposal Network (RPN) takes an image
# (of any size) as input and outputs a set of rectangular
# 3 object proposals, each with an objectness score. We
# model this process with a fully convolutional network [7], 
# which we describe in this section. Because our ultimate goal
# is to share computation with a Fast R-CNN object detection network [2], 
# we assume that both nets share a common set of convolutional layers.
# In our ex- periments, we investigate the Zeiler and Fergus model [32] (ZF), 
# which has 5 shareable convolutional layers and the Simonyan and Zisserman model [3] 
# (VGG-16), which has 13 shareable convolutional layers.

def common_layers_5(input_layer):
	'''
	Arguments:
		input_layer: image input layer
	Return:
		last_common_layer: last sharable convolutional layer
	'''
	conv_layer1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu',
		padding='same', data_format='channels_last')(input_layer)

	max_pooling_layer2 = MaxPooling2D(pool_size=(2,2),
		padding='same', data_format='channels_last')(conv_layer1)

	conv_layer3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu',
		padding='same', data_format='channels_last')(max_pooling_layer2)

	max_pooling_layer4 = MaxPooling2D(pool_size=(2,2),
		padding='same', data_format='channels_last')(conv_layer3)

	conv_layer5 = Conv2D(filters=64, kernel_size=(3,3), activation='relu',
		padding='same', data_format='channels_last')(max_pooling_layer4)

	return conv_layer5 # shape (16,16,64)


