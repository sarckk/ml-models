from trees.decision_tree import DecisionTree
from utils.exceptions import NotFittedError, InvalidArgError
from sklearn.ensemble import AdaBoostClassifier as AC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import numpy as np 
import numbers

NUMERICAL_STABILITY_CONSTANT = 1e-7

class AdaBoostClassifier:
	'''
		Implements Adaptive Boosting for Classification Tasks
	'''

	def __init__(self, base_estimator = DecisionTree, learning_rate = 1, n_estimators = 50):
		self.base_estimator = base_estimator 
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.estimators = []
		self.estimator_weights = []

	def _make_estimator(self, *args, **kwargs):
		estimator = self.base_estimator(*args, **kwargs)
		self.estimators.append(estimator)
		return estimator
	
	def _validate_sample_weight(self, sample_weight, n_samples):
		if sample_weight is None:
			sample_weight = np.ones(n_samples, dtype=np.float64)
		elif isinstance(sample_weight, numbers.Number):
			sample_weight = np.full(n_samples, sample_weight)
		else:
			sample_weight = np.array(sample_weight)
			if sample_weight.ndim != 1:
				raise InvalidArgError("list-like object passed as sample_weight must be 1-dimensional")
			if sample_weight.shape != (n_samples,):
				raise InvalidArgError(f"Expect sample_weight to have {n_samples} elements, but got {sample_weight.shape[0]} elements instead")

		sample_weight /= sample_weight.sum()
		if np.any(sample_weight < 0):
			raise InvalidArgError("sample_weight cannot have negative weights")
		return sample_weight

	def fit(self, X, y, sample_weight = None):
		'''
			AdaBoost implementation for non-multiclass output
		'''
		self.n_classes = len(np.unique(y))
		n_samples, n_classes = X.shape

		sample_weight = self._validate_sample_weight(sample_weight, n_samples)
		
		for i in range(self.n_estimators):
			stump = self._make_estimator(max_depth = 1)
			stump.fit(X, y, sample_weight)
			self.estimators.append(stump)

			pred_y = stump.predict(X)
			incorrect = pred_y != y

			total_error = sample_weight[incorrect].sum()
			# mult-class output would look like this:
			# error = np.mean(np.average(incorrect, weights=sample_weight, axis = 0))

			if total_error <= 0:
				break

			error_ratio  = (1-total_error) / total_error
			say = self.learning_rate * np.log(error_ratio + NUMERICAL_STABILITY_CONSTANT)
			self.estimator_weights.append(say)

			if i < self.n_estimators - 1:
				# If not last boost iteration, update and normalize
				sample_weight *= np.exp(say * incorrect)
				sample_weight /= sample_weight.sum()
		
	def predict(self, X):
		if len(self.estimators) == 0: 
			raise NotFittedError("Estimator not fitted yet. Call'fit' on the data before invoking 'predict'")

		# for each sample, do sum up the weights of the estimators that predicts a certain class for each class
		# then take the class with the max. say 
		n_samples = X.shape[0]
		weight_per_class = np.zeros((n_samples, self.n_classes)) 
		indices = np.arange(n_samples)
		for weight, estimator in zip(self.estimator_weights, self.estimators):
			weight_per_class[indices, estimator.predict(X)] += weight

		return weight_per_class.argmax(axis = 1)
	

if __name__ == '__main__':
	X, y = make_moons(n_samples=10000, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf = AC()
	clf.fit(X_train, y_train)
	clf_preds = clf.predict(X_test)
	print("Sklearn: ", accuracy_score(clf_preds, y_test))

	ours = AdaBoostClassifier(base_estimator=DecisionTreeClassifier)
	ours.fit(X_train, y_train)
	ours_preds = ours.predict(X_test)
	print("Ours: ", accuracy_score(ours_preds, y_test))