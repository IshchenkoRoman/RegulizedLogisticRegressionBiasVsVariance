import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

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
		# Here was problem just add reshape instead ".T"
		hypotesis = np.dot(theta, X.T).reshape(-1, 1)
		diff = hypotesis - y
		r = np.dot(diff.T, X)
		grad = r / m

		theta = theta.reshape(-1, 1)
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
	# 	reurn (cost)		

	def optimize(self, theta, X, y, lambd):
		
		print(theta.shape)
		op = minimize(self.linearRegCostFunct, x0=theta, args=(X, y, lambd), method=None, jac=self.gradient, options={'maxiter':200})

		return (op)

	def scikitOptimize(self, X, y):

		regr = LinearRegression(fir_intercept=False)
		regr.fit(X, y.ravel())

		cost = linearRegCostFunction(regr.coef_, X, y, 0)
		return (regr.coef_, cost)

	def plotData(self, X, y, xlabel='x', ylabel='y', label="None", option=1, line_x=None, line_y=None):

		fig, axs = plt.subplots(option, 1, squeeze=False)
		axs[0, 0].scatter(X, y, marker='x', c='r')
		axs[0, 0].set_xlim(-50, 60)
		axs[0, 0].set_ylim(-5, 40)
		axs[0, 0].set_xlabel(xlabel)
		axs[0, 0].set_ylabel(ylabel)
		axs[0, 0].grid(True)
		if (option == 2):
			axs[1, 0].scatter(X, y, marker='x', c='r')
			axs[1, 0].plot(line_x, line_y)
			axs[1, 0].set_xlim(-50, 60)
			axs[1, 0].set_ylim(-5, 40)
			axs[1, 0].set_xlabel(xlabel)
			axs[1, 0].set_ylabel(ylabel)
			axs[1, 0].grid(True)
		plt.tight_layout()
		plt.show()

def main():
	
	path_data = os.getcwd() + '/ex5data1.mat'
	rlr = RLRegr(path_data)

	# rlr.plotData(rlr.X[:,1], rlr.y, "Change in weather level (x)", "Wather flowing out of the dam (y)")
	theta = np.array([[15],[15]])
	# theta = np.zeros((rlr._yshape[0], 1)).reshape(1,12)
	# print(rlr.linearRegCostFunct(theta, rlr.X, rlr.y, 1))
	# print(rlr.gradient(theta, rlr.X, rlr.y, 1))
	op = rlr.optimize(theta, rlr.X, rlr.y, 0)
	xx = op.x
	print(op.x)
	print(rlr.X)
	# print(op.x.shape, rlr.X.shape)
	rlr.plotData(rlr.X[:,1], rlr.y, "Change in weather level (x)", "Wather flowing out of the dam (y)", label='Scipy optimize', option=2, line_x=np.linspace(-50, 40), line_y=(op.x[0] + (op.x[1] * np.linspace(-50, 40))))

if __name__ == '__main__':
	main()