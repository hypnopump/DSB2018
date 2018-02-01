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
	

def u_net(shape=(256, 256, 3), start=64, act="relu", dropout=None):
	""" Simple U-Net model from the original paper:
		https://arxiv.org/pdf/1505.04597.pdf

		Will try future improved versions such as:
			- https://arxiv.org/pdf/1705.03820v3.pdf
			- 
	"""
	inputs = Input(shape=shape)
	# First block
	conv1 = Conv2D(start, (3,3), padding="same", activation=act)(inputs)
	if dropout: conv1 = Dropout(dropout)(conv1)
	conv1 = Conv2D(start, (3,3), padding="same", activation=act)(conv1)
	# Second block
	conv2 = MaxPooling2D()(conv1)
	conv2 = Conv2D(start*2, (3,3), padding="same", activation=act)(conv2)
	if dropout: conv2 = Dropout(dropout)(conv2)
	conv2 = Conv2D(start*2, (3,3), padding="same", activation=act)(conv2)
	# Third block
	conv3 = MaxPooling2D()(conv2)
	conv3 = Conv2D(start*2*2, (3,3), padding="same", activation=act)(conv3)
	if dropout: conv3 = Dropout(dropout)(conv3)
	conv3 = Conv2D(start*2*2, (3,3), padding="same", activation=act)(conv3)

	# Fourth/last block
	conv4 = MaxPooling2D()(conv3)
	conv4 = Conv2D(start*2*2*2, (3,3), padding="same", activation=act)(conv4)
	if dropout: conv4 = Dropout(dropout)(conv4)
	conv4 = Conv2D(start*2*2*2, (3,3), padding="same", activation=act)(conv4)

	# STOP DOWNSAMPLING - START UPSAMPLING
	# Reverse third + Concatenate - (up-conv 2x2) as described in the original paper
	conv3_r = Conv2DTranspose(start*2*2, (2, 2), strides=(2, 2), padding='same')(conv4)
	conv3_r = Concatenate(axis=3)([conv3, conv3_r])
	# Normal part
	conv3_r = Conv2D(start*2*2, (3,3), padding="same", activation=act)(conv3_r)
	if dropout: conv3_r = Dropout(dropout)(conv3_r)
	conv3_r = Conv2D(start*2*2, (3,3), padding="same", activation=act)(conv3_r)

	# Reverse second + Concatenate - (up-conv 2x2) as described in the original paper
	conv2_r = Conv2DTranspose(start*2, (2, 2), strides=(2, 2), padding='same')(conv3_r)
	conv2_r = Concatenate(axis=3)([conv2, conv2_r])
	# Normal part
	conv2_r = Conv2D(start*2, (3,3), padding="same", activation=act)(conv2_r)
	if dropout: conv2_r = Dropout(dropout)(conv2_r)
	conv2_r = Conv2D(start*2, (3,3), padding="same", activation=act)(conv2_r)

	# Reverse first + Concatenate - (up-conv 2x2) as described in the original paper
	conv1_r = Conv2DTranspose(start, (2, 2), strides=(2, 2), padding='same')(conv2_r)
	conv1_r = Concatenate(axis=3)([conv1, conv1_r])
	# Normal part
	conv1_r = Conv2D(start, (3,3), padding="same", activation=act)(conv1_r)
	if dropout: conv1_r = Dropout(dropout)(conv1_r)
	conv1_r = Conv2D(start, (3,3), padding="same", activation=act)(conv1_r)

	conv1_r = Conv2D(1, (1,1), padding="same", activation="sigmoid")(conv1_r)
	outputs = Activation("sigmoid")(conv1_r)
	model = Model(inputs=inputs, outputs=outputs)
	return model 


if __name__ == "__main__":
	model = u_net(start=16)
	model.summary()