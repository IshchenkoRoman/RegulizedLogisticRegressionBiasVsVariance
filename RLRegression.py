import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.io import loadmat
from scipy.optimize import minimize

class RLRegr():

	def __init__(self, path_data):
		
		try:
			self._df = loadmat(path_data)
		except IOError:
			print("No data file!\n")
			raise

		self.y = self._df['y']
		self.yval = self._df['yval']
		self.Xval = self._df['Xval']
		self.Xtest = self._df['Xtest']
		self.X = self._df['X']
		self.ytest = self._df['ytest']
		self._Xshape = self.X.shape
		self._yshape = self.y.shape

		# add bias parametr
		self.X = np.c_[np.ones((self._Xshape[0], 1)), self.X]
		self._Xshape = self.X.shape

	def sigmoid(self, data):

		return (1 / (np.exp(-data)))

	# def gradient(self, theta, X, y, lambd):

	# 	m = y.size

	# 	h = X.dot(theta.reshape(-1, 1))

	# 	grad = (1 / m) * (X.T.dot(h-y)) + (lambd/m)*np.r_[[[0]], theta[1:].reshape(-1, 1)]

	# 	return (grad.flatten())

	def gradient(self, theta, X, y, lambd):

		m = len(y)
		# Here was problem just add reshape
		hypotesis = np.dot(theta, X.T).reshape(-1, 1)
		diff = hypotesis - y
		r = np.dot(diff.T, X)
		grad = r / m

		# print(theta.shape)
		theta = theta.reshape(-1, 1)
		# print(theta[1])
		t = ((lambd / m) * theta[1:])
		for i in range(1, grad.shape[1]):
			grad[:,i] = grad[:,i] + t
		return (grad.flatten())

	def linearRegCostFunct(self, theta, X, y, lambd):

		m = len(y)
		hypotesis = np.dot(theta, X.T).T
		diff = hypotesis - y
		summ = np.sum(diff ** 2) / (2 * m)

		l = ((lambd) / (2 * m)) * (np.sum(theta[1:]) ** 2)

		err = summ + l
		return (err)

	# def linearRegCostFunct(self, theta, X, y, lambd):

	# 	m = y.size

	# 	h = X.dot(theta)

	# 	cost = (1 / (2 * m)) * np.sum(np.square(h-y)) + (lambd / (2 * m)) * np.sum(np.square(theta[1:]))

	# 	return (cost)		

	def optimize(self, theta, X, y, lambd):
		
		# theta = np.array([[0], [0]])
		print(theta.shape)
		op = minimize(self.linearRegCostFunct, x0=theta, args=(X, y, lambd), method=None, jac=self.gradient, options={'maxiter':200})

		return (op)

	def plotData(self, X, y, xlabel='x', ylabel='y'):

		# fig = plt.figure(figsize=(640,480), facecolor='r')
		plt.scatter(X, y, marker='x', c='r')
		plt.xlim(-50, 40)
		plt.ylim(0, 40)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

def main():
	
	path_data = os.getcwd() + '/ex5data1.mat'
	rlr = RLRegr(path_data)

	# rlr.plotData(rlr.X[:,1], rlr.y, "Change in weather level (x)", "Wather flowing out of the dam (y)")
	theta = np.array([[15],[15]])
	# theta = np.zeros((rlr._yshape[0], 1)).reshape(1,12)
	# print(rlr.linearRegCostFunct(theta, rlr.X, rlr.y, 1))
	# print(rlr.gradient(theta, rlr.X, rlr.y, 1))
	print(rlr.optimize(theta, rlr.X, rlr.y, 0))


if __name__ == '__main__':
	main()