"""
    Simple Gradient Boosting Modules w/o support for multi-class output
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import _gb
from sklearn.model_selection import train_test_split
import numpy as np

class LeastSquaresError:
    def __call__(self, y, preds):
        return 1/2 * np.mean((y - preds) ** 2)

    def negative_gradient(self, y, prediction):
        return y - prediction

class BinomialError:
    def __call__(self, y, preds):
        pass

    def negative_gradient(self, y, prediction):
        pass



class BaseGradientBoosting:
    _losses = {
        'ls': LeastSquaresError,
    }

    def __init__(self, n_estimators, max_depth, learning_rate, min_samples_split, min_impurity_decrease, loss):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.loss = loss
    
    def _validate_params(self):
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError(f"Expected learning rate to be in (0.0,1.0) but instead got {self.learning_rate}.")
        
        if self.loss not in self._LOSS_CHOICES:
            raise ValueError(f"Loss function {self.loss} not supported.")
        
        if self.loss == 'deviance':
            self.loss_class = BinomialError 
        else:
            self.loss_class = self._losses[self.loss]
        
        self.loss_ = self.loss_class()

    def fit(self, X, y, sample_weight = None):
        self.n_samples, self.n_classes = X.shape

        # Validate params and instantiate loss class
        self._validate_params()

        # Initialize state including self.estimators_
        self.estimators_ = np.empty(self.n_estimators, dtype=object)
        self.train_score_ = np.zeros(self.n_estimators) 

        # Step 1 -- Get initial predictions
        self.init_pred = np.mean(y)
        predictions = self.init_pred

        # Step 2 -- Main loop
        for i in range(self.n_estimators):
            # Find negative of derivative of loss function w.r.t predictions (pseudo-residual)
            residual = self.loss_.negative_gradient(y, predictions)

            # Fit new decision tree on the residuals calculated above and create terminal regions
            estimator = DecisionTreeRegressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split, 
                                        min_impurity_decrease = self.min_impurity_decrease)
            estimator.fit(X, residual, sample_weight)

            # Get new predictions 
            predictions += self.learning_rate * estimator.predict(X)

            self.estimators_[i] = estimator
            self.train_score_[i] = self.loss_(y, predictions)


class GradientBoostingClassifier(BaseGradientBoosting):
    _LOSS_CHOICES = ['deviance']

    def __init__(self):
        super().__init__()
    
    def predict(self, X):
        pass
    
class GradientBoostingRegressor(BaseGradientBoosting):
    _LOSS_CHOICES = ['ls'] #, 'lad']

    def __init__(self, n_estimators = 100, max_depth = 3, learning_rate = 0.1, min_samples_split = 2, 
                min_impurity_decrease = 0.0, loss='ls'):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, 
            learning_rate=learning_rate, min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease, loss=loss)

    def predict(self, X):
        initial_preds = np.full(X.shape[0], self.init_pred)
        preds = [self.learning_rate * estimator.predict(X) for estimator in self.estimators_] 
        return initial_preds + np.sum(preds, axis = 0)
    
if __name__ == "__main__":
    X, y = make_regression(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    sk_gbr = _gb.GradientBoostingRegressor()
    sk_gbr.fit(X_train, y_train)
    print("Sklearn: ", mean_squared_error(sk_gbr.predict(X_test), y_test))

    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    print("Ours: ", mean_squared_error(gbr.predict(X_test), y_test))
