import keras
import numpy as np 
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Activation, Add, Concatenate, Dense
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose

def IoU_np(y, y_pred, t=0.5):
	""" Inputs two numpy arrays and outputs the Intersection over Union
		loss. IoU = (A^B)/(AvB)
	"""
	# Convert to 1 over treshold, 0 under treshold
	y_pred[y_pred > t] = 1
	y_pred[y_pred < t] = 0

	# Compute median across all tresholds from 0.5 to 0.95
	t=0
	for i in np.arange(0.5, 1.0, 0.05): # IoU computation
		t += np.sum(np.logical_and(y, y_pred))/np.sum(np.logical_or(y, y_pred))

	return t/np.sum(np.arange(0.5, 1.0, 0.05))

def IoU(y, y_pred, t=0.5):
	""" Inputs two keras tensors and outputs the Intersection over Union
		loss. IoU = (A^B)/(AvB)
		K.cumsum() = np.sum
		K.round = round to closest integer
		K.equal?
		keras.backend.switch(condition, then_expression, else_expression)
		K.mean?
	"""
	# Convert to 1 over treshold, 0 under treshold
	y_pred = K.switch(y_pred>=t, 1, 0)

	# Compute median across all tresholds from 0.5 to 0.95
	t=0
	for i in np.arange(0.5, 1.0, 0.05): # IoU computation
		t += np.sum(np.logical_and(y, y_pred))/np.sum(np.logical_or(y, y_pred))

	return t/np.sum(np.arange(0.5, 1.0, 0.05))

def u_net(shape=(256, 256, 3), start=64, act="relu"):
	""" Simple U-Net model from the original paper:
		https://arxiv.org/pdf/1505.04597.pdf

		Will try future improved versions such as:
			- https://arxiv.org/pdf/1705.03820v3.pdf
			- 
	"""
	inputs = Input(shape=shape)
	# First block
	first = Conv2D(start, (3,3), padding="same", activation=act)(inputs)
	first = Conv2D(start, (3,3), padding="same", activation=act)(first)
	# Second block
	second = MaxPooling2D()(first)
	second = Conv2D(start*2, (3,3), padding="same", activation=act)(second)
	second = Conv2D(start*2, (3,3), padding="same", activation=act)(second)
	# Third block
	third = MaxPooling2D()(second)
	third = Conv2D(start*2*2, (3,3), padding="same", activation=act)(third)
	third = Conv2D(start*2*2, (3,3), padding="same", activation=act)(third)
	# Fourth/last block
	fourth = MaxPooling2D()(third)
	fourth = Conv2D(start*2*2*2, (3,3), padding="same", activation=act)(fourth)
	fourth = Conv2D(start*2*2*2, (3,3), padding="same", activation=act)(fourth)

	# STOP DOWNSAMPLING - START UPSAMPLING
	# Reverse third - (up-conv 2x2) as described in the original paper
	r_third = Conv2DTranspose(start*2*2, (2, 2), strides=(2, 2), padding='same')(fourth)
	# Concatenate
	r_third = Concatenate(axis=3)([third, r_third])
	# Normal part
	r_third = Conv2D(start*2*2, (3,3), padding="same", activation=act)(r_third)
	r_third = Conv2D(start*2*2, (3,3), padding="same", activation=act)(r_third)

	# Reverse second
	r_second = Conv2DTranspose(start*2, (2, 2), strides=(2, 2), padding='same')(r_third)
	# Concatenate
	r_second = Concatenate(axis=3)([second, r_second])
	# Normal part
	r_second = Conv2D(start*2, (3,3), padding="same", activation=act)(r_second)
	r_second = Conv2D(start*2, (3,3), padding="same", activation=act)(r_second)

	# Reverse first
	r_first = Conv2DTranspose(start, (2, 2), strides=(2, 2), padding='same')(r_second)
	# Concatenate
	r_first = Concatenate(axis=3)([first, r_first])
	# Normal part
	r_first = Conv2D(start, (3,3), padding="same", activation=act)(r_first)
	r_first = Conv2D(start, (3,3), padding="same", activation=act)(r_first)

	r_first = Conv2D(1, (1,1), padding="same", activation="sigmoid")(r_first)
	outputs = Activation("sigmoid")(r_first)
	model = Model(inputs=inputs, outputs=outputs)
	return model 


if __name__ == "__main__":
	model = u_net()
	model.summary()