from numpy import array, sqrt
from warnings import warn

def euclidean_distance(x, y):
	x = array(x).reshape(1, -1)
	y = array(y).reshape(1, -1)

	if x.shape != y.shape:
		warn(UserWarning("Using sequences which do not contain equivalent numbers of items can result in unexpected results."))

	return sqrt(sum((x - y)**2 for x, y in zip(x[0], y[0])))
 