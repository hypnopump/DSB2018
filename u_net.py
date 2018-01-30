import keras
import numpy as np 
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Activation, Add, Concatenate, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D

def IoU(y, y_pred, t=0.5):
	""" Inputs two numpy arrays and outputs the Intersection over Union
		loss. IoU = (A^B)/(AvB)
	"""
	# Convert to 1 over treshold, 0 under treshold
	y_pred[y_pred > t] = 1
	y_pred[y_pred < t] = 0
	# IoU computation
	return np.sum(np.logical_and(y, y_pred))/np.sum(np.logical_or(y, y_pred))

def u_net(shape=(256, 256, 3), start=64, act="relu"):
	""" Simple U-Net model from the original paper:
		https://arxiv.org/pdf/1505.04597.pdf

		Will try future improved versions such as:
			- https://arxiv.org/pdf/1705.03820v3.pdf
			- 
	"""
	inputs = Input(shape=shape)
	# First block
	first = Conv2D(start, (3,3), padding="same")(inputs)
	first = Activation(act)(first)
	first = Conv2D(start, (3,3), padding="same")(first)
	first = Activation(act)(first)
	# Second block
	second = MaxPooling2D()(first)
	second = Conv2D(start*2, (3,3), padding="same")(second)
	second = Activation(act)(second)
	second = Conv2D(start*2, (3,3), padding="same")(second)
	second  = Activation(act)(second)
	# Third block
	third = MaxPooling2D()(second)
	third = Conv2D(start*2*2, (3,3), padding="same")(third)
	third = Activation(act)(third)
	third = Conv2D(start*2*2, (3,3), padding="same")(third)
	third  = Activation(act)(third)
	# Fourth/last block
	fourth = MaxPooling2D()(third)
	fourth = Conv2D(start*2*2*2, (3,3), padding="same")(fourth)
	fourth = Activation(act)(fourth)
	fourth = Conv2D(start*2*2*2, (3,3), padding="same")(fourth)
	fourth  = Activation(act)(fourth)

	# STOP DOWNSAMPLING - START UPSAMPLING
	# Reverse third - Upsampling conv (up-conv 2x2) as described in
	# the original paper
	r_third = UpSampling2D()(fourth)
	r_third = Conv2D(start*2*2, (2,2), padding="same")(r_third)
	# Concatenate
	r_third = Concatenate(axis=0)([third, r_third])
	# Normal part
	r_third = Conv2D(start*2*2, (3,3), padding="same")(r_third)
	r_third = Activation(act)(r_third)
	r_third = Conv2D(start*2*2, (3,3), padding="same")(r_third)
	r_third  = Activation(act)(r_third)

	# Reverse second
	r_second = UpSampling2D()(r_third)
	r_second = Conv2D(start*2, (2,2), padding="same")(r_second)
	# Concatenate
	r_second = Concatenate(axis=0)([second, r_second])
	# Normal part
	r_second = Conv2D(start*2, (3,3), padding="same")(r_second)
	r_second = Activation(act)(r_second)
	r_second = Conv2D(start*2, (3,3), padding="same")(r_second)
	r_second  = Activation(act)(r_second)

	# Reverse first
	r_first = UpSampling2D()(r_second)
	r_first = Conv2D(start, (2,2), padding="same")(r_first)
	# Concatenate
	r_first = Concatenate(axis=0)([first, r_first])
	# Normal part
	r_first = Conv2D(start, (3,3), padding="same")(r_first)
	r_first = Activation(act)(r_first)
	r_first = Conv2D(start, (3,3), padding="same")(r_first)
	r_first  = Activation(act)(r_first)

	outputs = Conv2D(3, (1,1), padding="same", activation="sigmoid")(r_first)
	model = Model(inputs=inputs, outputs=outputs)
	return model 


if __name__ == "__main__":
	model = u_net()
	model.summary()