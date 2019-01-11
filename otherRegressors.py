import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer



# regressions :


#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, SGDRegressor, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb
#import lightgbm as lgb
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

X_cleaned=pd.read_csv('X_cleaned.csv')
X_cleaned=pd.DataFrame(X_cleaned)
X1=pd.read_csv('X1.csv')
X1=pd.DataFrame(X1)
Y1=pd.read_csv('Y1.csv')
Y1=pd.DataFrame(Y1)
train=pd.read_csv('train.csv')
train=pd.DataFrame(train)
test=pd.read_csv('test.csv')
test=pd.DataFrame(test)

print(Y1.shape)
print(X_cleaned.shape)
print(X1.shape)
print(train.shape)
print(test.shape)

for i in train:
	print(i)

print(train.head(6))
print(test.head(6))





