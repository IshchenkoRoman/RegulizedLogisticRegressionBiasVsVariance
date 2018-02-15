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
		self.polyX = None

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

	def myGradient(self, theta, X, y, lambd):

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

	def myLinearRegCostFunct(self, theta, X, y, lambd):

		m = len(y)
		hypotesis = np.dot(theta, X.T).T
		diff = hypotesis - y
		summ = np.sum(diff ** 2) / (2 * m)

		l = ((lambd) / (2 * m)) * (np.sum(theta[1:]) ** 2)

		err = summ + l
		return (err)

	def linearRegCostFunct(self, theta, X, y, lambd, return_grad=False):

		m = len(y)

		theta = np.reshape(theta, (-1, y.shape[1]))

		J = 0
		grad = np.zeros(theta.shape)

		J = (1./(2*m)) * np.power((np.dot(X, theta) - y), 2).sum() + (float(lambd) / (2*m)) * np.power(theta[1:theta.shape[0]], 2).sum()

		if return_grad == True:

			grad = (1./m) * np.dot(X.T, np.dot(X, theta) - y) + (float(lambd) / m)*theta
			grad_no_regularization = (1./m) * np.dot(X.T, np.dot(X, theta) - y)
			grad[0] = grad_no_regularization[0]

			return J, grad.flatten()
		else:
			return J

	# def linearRegCostFunct(self, theta, X, y, lambd):

	# 	m = y.size
	# 	h = X.dot(theta)
	# 	cost = (1 / (2 * m)) * np.sum(np.square(h-y)) + (lambd / (2 * m)) * np.sum(np.square(theta[1:]))
	# 	reurn (cost)		

	def optimizeWithMyGradient(self, theta1, X, y, lambd):

		theta = np.zeros((X.shape[1], 1))
		op = minimize(self.myLinearRegCostFunct, x0=theta, args=(X, y, lambd), method=None, jac=self.myGradient, options={'maxiter':200})
		return (op)

	def optimizeWithLBFGSB(self, X, y, lambd):
		
		theta = np.zeros((X.shape[1], 1))
		op = minimize(self.linearRegCostFunct, x0=theta, args=(X, y, lambd, True), method='L-BFGS-B', jac=True, options={'disp':True, 'maxiter':200})
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
		if (option == 2 and line_x != None and line_y != None):
			axs[1, 0].scatter(X, y, marker='x', c='r')
			axs[1, 0].plot(line_x, line_y)
			axs[1, 0].set_xlim(-50, 60)
			axs[1, 0].set_ylim(-5, 40)
			axs[1, 0].set_xlabel(xlabel)
			axs[1, 0].set_ylabel(ylabel)
			axs[1, 0].grid(True)
		plt.tight_layout()
		plt.show()

	def learningCurves(self, X, y, Xval, yval, lambd):

		m = y.size

		validateX = Xval
		validatey = yval

		array_of_train_error = []
		array_of_validate_error = []
		theta_opt = np.zeros((X.shape[1], 1))
		for i in range(0, m):
			optimize = self.optimizeWithLBFGSB(X[:i+1], y[:i+1], lambd)
			# print(optimize)
			array_of_train_error.append(self.linearRegCostFunct(optimize.x, X[:i+1], y[:i+1], 0))
			array_of_validate_error.append(self.linearRegCostFunct(optimize.x, validateX, validatey, 0))

		return (array_of_train_error, array_of_validate_error)

	def plotCurvesHighBias(self, err_train, err_val):

		number_of_training = np.arange(1, 13)

		et = plt.plot(number_of_training, err_train)
		ev = plt.plot(number_of_training, err_val)
		plt.xlim(0, 12)
		plt.ylim(0, max(err_val) + max(err_val) % 10)
		plt.xlabel("Number of training examples")
		plt.ylabel("Error")
		plt.legend([et[0], ev[0]], ["Train", "Cross Validation"])

		plt.show()

	def createPolynomX(self, power, Xtrain):

		# poly = PolynomialFeatures(degree=power)
		X_res = np.ones((Xtrain.shape[0], power))
		y, x = X_res.shape

		for i in range(y):
			for j in range(x):
				X_res[i][j] = Xtrain[i] ** (j + 1)

		return (X_res)

	def featureNormilize(self, polyX):

		mu = np.mean(polyX, axis=0)
		X_norm = polyX - mu

		sigma = np.std(X_norm, axis=0)
		X_norm = X_norm/sigma

		return (X_norm, mu, sigma)

	def _plotFit(self, min_x, max_x, mu, sigma, theta, p, left=15, right=25):
		
		x = np.array(np.arange(min_x - left, max_x + right, 0.05))

		polynom = self.createPolynomX(p, x)
		polynom = (polynom - mu) / sigma
		polynom = np.c_[np.ones((polynom.shape[0], 1)), polynom]

		# print(theta.shape, polynom.shape)
		pol = plt.plot(x, np.dot(polynom, theta), '-', linewidth=2)
		return (pol)

	def plotPolyReg(self, polyX, y, mu, sigma, p, lambd=1):

		init_theta = np.zeros((polyX.shape[1], 1))
		theta = self.optimizeWithLBFGSB(polyX, y, lambd)

		data = plt.plot(self.X[:,1], y, 'rx', markersize=7, mew=2)
		pol = self._plotFit(min(self.X[:,1]), max(self.X[:,1]), mu, sigma, theta.x, p)
		# plt.xlim(-80, 60)
		# plt.ylim(min(self.X[:,1]) - min(self.X[:,1]) % 10, max(self.X[:,1]) + max(self.X[:,1]) % 10)
		plt.xlabel("Change in water level (x)", fontsize=10)
		plt.ylabel("Water flowing out of the dam (y)", fontsize=10)
		plt.legend([data[0], pol[0]], ["Our data", "Polynomial curve of {0} degree".format(p)])
		plt.title("Polynomial Regression Fit (lambda = {:f})".format(lambd), fontsize=10)

		plt.show()

	def plotPolyLearningCurves(self, Xpoly, y, Xval, yval, lambd):

		plt.close()
		m = y.size
		# plt.figure(figsize=(12,8))

		err_train, err_val =self.learningCurves(Xpoly, y, Xval, yval, lambd)
		p1, p2 = plt.plot(range(1, m+1), err_train, range(1, m+1), err_val, linewidth=2)

		plt.axis([0, 13, 0, 100])
		legend = plt.legend((p1, p2), ("Traine", "Validation"), fontsize=10)
		for label in legend.get_lines():
			label.set_linewidth(3)

		plt.title("Polynomial Regression Learning Curve (lambda = {:f})".format(lambd), fontsize=10)
		plt.xlabel("Number of training examples", fontsize=10)
		plt.ylabel("Error")

		plt.show()

		print("Polynomial Regression (lambda = {:f})\n\n".format(lambd))
		print("# Training Examples\tTrain Error\tValidation Error\n")
		for i in range(m):
			print("  \t{:d}\t\t{:f}\t{:f}\n".format(i+1, float(err_train[i]), float(err_val[i])))

	def _validationCurve(self, X, y, Xval, yval):

		lambdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
		len_lambdas = lambdas.size

		err_train = np.zeros((1, len_lambdas))[0]
		err_val = np.zeros((1, len_lambdas))[0]
		theta = 0
		print(err_val)
		# theta_array = np.zeros((1, len_lambdas))

		for i in range(len_lambdas):

			theta = self.optimizeWithLBFGSB(X, y, lambdas[i])
			err_train[i] = self.linearRegCostFunct(theta.x, X, y, 0)
			err_val[i] = self.linearRegCostFunct(theta.x, Xval, yval, 0)

		return (lambdas, err_train, err_val)

	def plotDifferentLambdasError(self, X, y, Xval, yval):

		lambdas_vec, err_train, err_val = self._validationCurve(X, y, Xval, yval)

		p1, p2 = plt.plot(lambdas_vec, err_train, lambdas_vec, err_val)
		legend = plt.legend((p1, p2), ("Train", "Validation"), fontsize=10)
		for label in legend.get_lines():
			label.set_linewidth(3)

		plt.xlabel(" Lambda ", fontsize=10)
		plt.ylabel(" Error ")
		# plt.axis([0, 10, 0, 20])

		plt.show()

		print("Lambda\t\tTrain Error\tValidation Error\n")
		for i in range(len(lambdas_vec)):
			print("  {:f}\t{:f}\t{:f}\n".format(lambdas_vec[i], err_train[i], err_val[i]))




def main():
	
	path_data = os.getcwd() + '/ex5data1.mat'
	rlr = RLRegr(path_data)

	"""
	First part of task
	"""

	# Draw our data of trining set
	# # rlr.plotData(rlr.X[:,1], rlr.y, "Change in weather level (x)", "Wather flowing out of the dam (y)")

	# theta = np.zeros((rlr.X.shape[1], 1))

	# print(rlr.linearRegCostFunct(theta, rlr.X, rlr.y, 1))
	# # # print(rlr.gradient(theta, rlr.X, rlr.y, 1))

	# # Train theta
	# op = rlr.optimizeWithLBFGSB(theta, rlr.X, rlr.y, 0)
	# # print(op)
	# # rlr.plotData(rlr.X[:,1], rlr.y, "Change in weather level (x)", "Wather flowing out of the dam (y)", label='Scipy optimize', option=2, line_x=np.linspace(-50, 40), line_y=(op.x[0] + (op.x[1] * np.linspace(-50, 40))))

	# # In Learning curves we make subsets of our training data and calculate errors and train theta for this subsets and then with that thetas cfalculate error whole Validate set
	# Xval = np.c_[np.ones((rlr.Xval.shape[0], 1)), rlr.Xval]
	# err_train, err_val = rlr.learningCurves(rlr.X, rlr.y, Xval, rlr.yval, 1)
	# # Plot dependence between size of subset and errors of train sbuset and validate set
	# rlr.plotCurvesHighBias(err_train, err_val)

	"""
	Second part, polynomial regression
	"""

	# So, we create our polynomial features, that can improve accurancy of prediction

	p = 8
	polynom = rlr.createPolynomX(p, rlr.X[:,1])

	polynom, mu, sigma = rlr.featureNormilize(polynom)
	polynom = np.c_[np.ones((polynom.shape[0], 1)), polynom]

	polynom_test = rlr.createPolynomX(p, rlr.Xtest)
	polynom_test = (polynom_test - mu) / sigma
	polynom_test = np.c_[np.ones((polynom_test.shape[0], 1)), polynom_test]

	polynom_val = rlr.createPolynomX(p, rlr.Xval)
	polynom_val = (polynom_val - mu) / sigma
	polynom_val = np.c_[np.ones((polynom_val.shape[0], 1)), polynom_val]

	# Here and below we can see how lambda is influence on accurancy of prediction

	# rlr.plotPolyReg(polynom, rlr.y, mu, sigma, p, 0)
	# rlr.plotPolyLearningCurves(polynom, rlr.y, polynom_val, rlr.yval, 0)

	# rlr.plotPolyReg(polynom, rlr.y, mu, sigma, p, 1)
	# rlr.plotPolyLearningCurves(polynom, rlr.y, polynom_val, rlr.yval, 1)

	# rlr.plotPolyReg(polynom, rlr.y, mu, sigma, p, 100)
	# rlr.plotPolyLearningCurves(polynom, rlr.y, polynom_val, rlr.yval, 100)

	# Plot different lambdas and print errors of lambdas, to understand which one is the best solution for our task

	# rlr.plotDifferentLambdasError(polynom, rlr.y, polynom_val, rlr.yval)

	"""

	Lambda		Train Error	Validation Error

	  0.000000	0.028890	53.919105

	  0.001000	0.107976	9.349919

	  0.003000	0.166731	15.920055

	  0.010000	0.217958	17.149315

	  0.030000	0.275149	13.216183

	  0.100000	0.438656	7.926826

	  0.300000	0.868209	4.760732

	  1.000000	1.958697	4.263453

	  3.000000	4.525101	3.832184

	  10.000000	14.825780	8.889697


	We see that our lambda has value nearly 3.832184
	"""

	lambd = 3
	theta = rlr.optimizeWithLBFGSB(polynom, rlr.y, lambd)

	err = rlr.linearRegCostFunct(theta.x, polynom_test, rlr.ytest, 0)
	print(err)



if __name__ == '__main__':
	main()