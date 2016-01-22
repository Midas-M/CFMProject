from __future__ import division

import pandas as pd
import os
from sklearn.preprocessing import scale, PolynomialFeatures,normalize
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.learning_curve import learning_curve
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts

class FeatureExtractor(object):

    def __init__(self):
        pass
    def fit(self, X_df, y_array):
       pass
    def transform(self, X_df):
        X_array=self.initFeatureSet(X_df)
        return X_array

    def initFeatureSet(self,X_df):
       pass


