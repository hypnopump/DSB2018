import keras
import numpy as np 
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Activation, Add, Concatenate, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, AveragePooling2D

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

# Metrics to train the network
# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
	

def u_net(shape=(256, 256, 3), start=64, act="relu", starter="he", dropout=None, pool="max"):
	""" Simple U-Net model from the original paper:
		https://arxiv.org/pdf/1505.04597.pdf

		Will try future improved versions such as:
			- https://arxiv.org/pdf/1705.03820v3.pdf
			- 
	"""
	initializer = {"he": "he_normal", "glorot": "glorot_uniform"}
	weight_decay = l2(0)
	pooler = {"max": MaxPooling2D(), "avg": AveragePooling2D()}
	inputs = Input(shape=shape)
	s = Lambda(lambda x: x / 255) (inputs)

	c1 = Conv2D(start, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (s)
	if dropout: c1 = Dropout(0.1) (c1)
	c1 = Conv2D(start, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c1)
	p1 = pooler[pool](c1)

	c2 = Conv2D(start*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (p1)
	if dropout: c2 = Dropout(0.1) (c2)
	c2 = Conv2D(start*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c2)
	p2 = pooler[pool] (c2)

	c3 = Conv2D(start*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (p2)
	if dropout: c3 = Dropout(0.2) (c3)
	c3 = Conv2D(start*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c3)
	p3 = pooler[pool] (c3)

	c4 = Conv2D(start*2*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (p3)
	if dropout: c4 = Dropout(0.2) (c4)
	c4 = Conv2D(start*2*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c4)
	p4 = pooler[pool] (c4)

	c5 = Conv2D(start*2*2*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (p4)
	if dropout: c5 = Dropout(0.3) (c5)
	c5 = Conv2D(start*2*2*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c5)

	u6 = Conv2DTranspose(start*2*2*2, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = Concatenate(axis=3)([u6, c4])
	c6 = Conv2D(start*2*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (u6)
	if dropout: c6 = Dropout(0.2) (c6)
	c6 = Conv2D(start*2*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c6)

	u7 = Conv2DTranspose(start*2*2, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = Concatenate()([u7, c3])
	c7 = Conv2D(start*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (u7)
	if dropout: c7 = Dropout(0.2) (c7)
	c7 = Conv2D(start*2*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c7)

	u8 = Conv2DTranspose(start*2, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = Concatenate(axis=3)([u8, c2])
	c8 = Conv2D(start*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (u8)
	if dropout: c8 = Dropout(0.1) (c8)
	c8 = Conv2D(start*2, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c8)

	u9 = Conv2DTranspose(start, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = Concatenate(axis=3)([u9, c1])
	c9 = Conv2D(start, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (u9)
	if dropout: c9 = Dropout(0.1) (c9)
	c9 = Conv2D(start, (3, 3), activation=act, kernel_initializer=initializer[starter], kernel_regularizer=weight_decay, padding='same') (c9)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

	model = Model(inputs=[inputs], outputs=[outputs])
	return model


if __name__ == "__main__":
	model = u_net(start=16)
	model.summary()