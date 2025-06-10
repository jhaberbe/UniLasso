import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LassoCV
from scipy.optimize import nnls

# Scikit-learn type interface
class UniLasso:
    def __init__(self):
        # Here we'll store the intercepts, coefficients, and fitted values for the univariate models.
        self.intercepts_ = None
        self.coef_ = None
        self.F_ = None

        # We use LassoCV to select the best regularization parameter.
        # We set positive=True to ensure that coefficients are non-negative.
        self.lasso = LassoCV(positive=True, alphas=100, cv=5, max_iter=10000, tol=1e-4, n_jobs=-1, fit_intercept=False)
    
    def fit(self, X, y):
        # Initialize the intercepts, coefficients, and fitted values matrix based on the input shape.
        self.intercepts_ = np.zeros(X.shape[1], dtype=float)
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.F_ = np.zeros_like(X, dtype=float)

        # Over all the features.
        for j in range(X.shape[1]):
            # Center the matrix
            xj = X[:, j]
            xj_centered = xj - xj.mean()
            hat_diag = xj_centered**2 / np.sum(xj_centered**2)

            # Get our linear regression fit.
            beta = np.sum(xj_centered * y) / np.sum(xj_centered**2)
            intercept = y.mean() - beta * xj.mean()

            # Generate predictions for the current feature.
            y_pred = intercept + xj * beta

            # Store the intercept and coefficient for predictions on unseen data.
            self.intercepts_[j] = intercept
            self.coef_[j] = beta

            # Update the fitted values matrix, this is a trick to avoid recomputing the entire system each time.
            self.F_[:, j] = y_pred - hat_diag * (y_pred - y) / (1 - hat_diag)

        self.lasso.fit(self.F_, y)
    
    def predict(self, X):
        # Rebuild the F matrix based on training-time univariate fits
        # This assumes we're working with unseen data data.
        F_new = X * self.coef_ + self.intercepts_  # elementwise per feature
        return self.lasso.predict(F_new)

    def score(self, y):
        # Rebuild the F matrix based on training-time univariate fits
        # This assumes we're working with unseen data data.
        return self.lasso.score(self.F_, y)

class IsotonicEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 increasing=False, 
                 out_of_bounds="clip", 
                 weighting="nnls", 
                 lasso_cv=5):
        """
        Parameters:
        - increasing: Whether isotonic regression should be increasing.
        - out_of_bounds: How to handle values outside training range ('clip' or 'nan').
        - weighting: 'nnls' or 'lasso' for combining predictions.
        - lasso_cv: Number of CV folds for LassoCV (used if weighting='lasso').
        """
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds
        self.weighting = weighting
        self.lasso_cv = lasso_cv
        
    def fit(self, X, y):
        X = pd.DataFrame(X)  # Ensure we can use iloc
        self.n_features_ = X.shape[1]
        
        # Fit isotonic models
        self.regressors_ = {
            i: IsotonicRegression(
                increasing=self.increasing, 
                out_of_bounds=self.out_of_bounds
            ).fit(X.iloc[:, i], y)
            for i in range(self.n_features_)
        }
        
        # Stack predictions
        self.predictions_matrix_ = np.column_stack([
            self.regressors_[i].predict(X.iloc[:, i]) 
            for i in range(self.n_features_)
        ])
        
        # Fit combiner
        if self.weighting == "nnls":
            self.weights_, _ = nnls(self.predictions_matrix_, y, maxiter=1_000_000)
        elif self.weighting == "lasso":
            model = LassoCV(cv=self.lasso_cv, fit_intercept=False, max_iter=10_000)
            model.fit(self.predictions_matrix_, y)
            self.weights_ = model.coef_
            self.alpha_ = model.alpha_
        else:
            raise ValueError("weighting must be either 'nnls' or 'lasso'")
        
        return self
    
    def predict(self, X):
        X = pd.DataFrame(X)
        pred_matrix = np.column_stack([
            self.regressors_[i].predict(X.iloc[:, i]) 
            for i in range(self.n_features_)
        ])
        return np.dot(pred_matrix, self.weights_)
