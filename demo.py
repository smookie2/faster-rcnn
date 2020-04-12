from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from common_layers import common_layers_5 as common_layers

img_shape = (64, 64, 3)

input_layer = Input(shape=img_shape)
common_layers = common_layers(input_layer)

rpn_model = Model(inputs=input_layer, outputs=common_layers, name="RPN")

rpn_model.compile(optimizer=Adam(lr=0.0001), loss='mse')
rpn_model.summary()