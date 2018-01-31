import numpy as np 
import time

def IoU(y, y_pred, t=0.5):
	""" Inputs two numpy arrays and outputs the Intersection over Union
		loss. IoU = (A^B)/(AvB)
	"""
	# Convert to 1 over treshold, 0 under treshold
	y_pred[y_pred > t] = 1
	y_pred[y_pred < t] = 0

	t=0
	for i in np.arange(0.5, 1.0, 0.05):
		# IoU computation
		t += np.sum(np.logical_and(y, y_pred))/np.sum(np.logical_or(y, y_pred))

	return t/np.sum(np.arange(0.5, 1.0, 0.05))


a = np.array([[0,0,0,0,1,1,1,1,1,0,0,0,0],
			  [0,0,0,0,0,1,1,1,1,1,0,0,0],
			  [0,0,1,1,1,1,1,1,0,0,0,0,0]])

b = np.array([[0,0,1,1,1,1,1,1,0,0,0,0,0],
			  [0,0,0,0,0,0,1,1,1,1,1,0,0],
			  [0,0,0,0,1,1,1,1,0,0,0,0,0]])

tac = time.time()
x = 25000
for i in range(x):
	print(IoU(a,b))
tic = time.time()

print("Time for {0} IoU: {1}".format(x, tic-tac))