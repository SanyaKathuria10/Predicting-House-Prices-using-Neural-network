import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler



#read train features

X=pd.read_csv('../data/X.csv')
# read train labels
Y=pd.read_csv('../data/Y.csv')
#convert them into dataframes
X=pd.DataFrame(X)

Y=pd.DataFrame(Y)
# drop the columns headings

#print("Shapes are ",X1.shape, Y1.shape)






def base_model1():
     model = Sequential()
     model.add(Dense(kernel_initializer="normal", units=14, activation="relu", input_dim=64))
     model.add(Dense(10, init='normal', activation='relu'))
     model.add(Dense(kernel_initializer="normal", units=2))
     model.compile(loss='mean_squared_error', optimizer = 'adam')
     return model
 
seed = 7
np.random.seed(seed)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)




#scale = StandardScaler()
#scale.fit(X_train)
#X_train = scale.transform(X_train)
#X_test = scale.transform(X_test)



clf_cleaned = KerasRegressor(build_fn=base_model1)
clf_cleaned.fit(X_train,y_train,epochs=50, batch_size=40)
#print(clf)
res_cleaned = clf_cleaned.predict(X_test)
## line below throws an error
#print(res_cleaned)





rmse1=np.sqrt(np.square(np.subtract(y_test,res_cleaned)).mean())
print("RMSE", rmse1)


































"""


# create a model
def baseline_model1():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal',activation='relu'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
	return model




seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=5, random_state=seed)
estimator_cleaned = KerasRegressor(build_fn=baseline_model1, epochs=10, batch_size=5, verbose=0)
results_cleaned = cross_val_score(estimator_cleaned, X_cleaned, Y1, cv=kfold)
print(results_cleaned)





def baseline_model2():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=76, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal',activation='relu'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
	return model




seed = 7
np.random.seed(seed)

estimator_X1 = KerasRegressor(build_fn=baseline_model2, epochs=10, batch_size=5, verbose=0)
results_X1 = cross_val_score(estimator_X1, X1, Y1, cv=kfold)

print(results_X1)





"""







