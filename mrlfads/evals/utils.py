import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

def flatten(arr): return arr.reshape(-1, arr.shape[-1])

class PolyRegression:
    def __init__(self, degree, alpha=0.0, tpe="ridge"):
        self.degree = degree
        self.alpha = alpha
        self.poly_features = PolynomialFeatures(degree=degree)
        
        if tpe == "ridge":
            self.reg = Ridge(alpha=alpha)
        elif tpe == "lasso":
            self.reg = Lasso(alpha=alpha)
        else:
            raise ValueError()

    def fit(self, X, y):
        X_poly = self.poly_features.fit_transform(X)
        self.reg.fit(X_poly, y)
        
    def ffit(self, X, y):
        self.fit(flatten(X), flatten(y))

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return self.reg.predict(X_poly)
    
    def fpredict(self, X):
        X_poly = self.poly_features.transform(flatten(X))
        return self.reg.predict(X_poly).reshape(*X.shape[:2], -1)

    def score(self, X, y): # X: prediction, y: true
        X_poly = self.poly_features.transform(X)
        return self.reg.score(X_poly, y)
    
    def fscore(self, X, y):
        pred = self.fpredict(X)
        return r2_score(flatten(y), flatten(pred))
    
