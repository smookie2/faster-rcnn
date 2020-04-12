from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense

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

	return conv_layer5


