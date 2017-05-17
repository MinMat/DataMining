
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import math


mypath = os.path.dirname(os.path.realpath('train.csv'))
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
os.path.exists(mypath)
print(onlyfiles[8])


#Read Training CSV for house regression
dataset = pd.read_csv(onlyfiles[8])
dataset.shape
dataset.drop(['Id'], axis=1)
dataset.head()

testDataset = pd.read_csv(onlyfiles[7])
testDataset.head()
testDataset.drop(['Id'], axis=1)

#Delete redundant features with over 50% NAN values E.G: porch sizes, pool area, alley, miscallenous features and its value 
unwantedFeaturesList = ["WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea",
                       "PoolQC","Fence","Alley","MiscFeature","MiscVal"]
dataset.drop(unwantedFeaturesList,inplace=True,axis=1)
testDataset.drop(unwantedFeaturesList,inplace=True,axis=1)

# Check all the columns fetures whcih has a numerical value type
is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
is_number(dataset.dtypes)

#Separate and select dataframes by dataTypes 
df_NonNumericalType = dataset.select_dtypes(exclude=[np.number])
df_NumericalType = dataset.select_dtypes(include=[np.number])

#prepare test dataset
df_TestNonNumericalType = testDataset.select_dtypes(exclude=[np.number])
df_TestNumericalType = testDataset.select_dtypes(include=[np.number])

#Set type to categorical
df_NonNumericalType[df_NonNumericalType.select_dtypes(['object']).columns] = df_NonNumericalType.select_dtypes(['object']).apply(lambda x: x.astype('category'))
df_TestNonNumericalType[df_TestNonNumericalType.select_dtypes(['object']).columns] = df_TestNonNumericalType.select_dtypes(['object']).apply(lambda x: x.astype('category'))


df_NumericalType.drop(["Id"], inplace=True, axis=1)
df_NumericalType.drop(['SalePrice'], inplace=True, axis=1)
df_TestNumericalType.drop(["Id"], inplace=True, axis=1)

#df_NumericalType.drop(["Id"], inplace=True, axis=1)
#replace all the NA values to numpy NAN for float 64 data type
df_NumericalType.replace('NA',np.nan)
df_TestNumericalType.replace('NA',np.nan)
#Select Numerical values except the index 
xNumerical = df_NumericalType.values
xTestNumerical = df_TestNumericalType.values

from sklearn.preprocessing import Imputer
if df_TestNumericalType.isnull().values.any() :
    #mean is the deafualt parameter anywyas
    imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
    imputer.fit(xTestNumerical[:,0:29])
    xTestNumerical[:,0:29] = imputer.transform(xTestNumerical[:,0:29])


from sklearn.preprocessing import Imputer
if df_NumericalType.isnull().values.any() :
    #mean is the deafualt parameter anywyas
    imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
    imputer.fit(xNumerical[:,0:28])
    xNumerical[:,0:28] = imputer.transform(xNumerical[:,0:28])
pd.DataFrame(xNumerical).head()



np.isnan(xTestNumerical).any()

#Scaling Numerical data
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
NumericalScaledData = sc_X.fit_transform(xNumerical)
TestNumericalScaledData = sc_X.fit_transform(xTestNumerical)


df_AllNonNumerical = pd.concat([df_NonNumericalType, df_TestNonNumericalType], axis=0)
df_AllNonNumerical[df_AllNonNumerical.select_dtypes(['object']).columns] = df_AllNonNumerical.select_dtypes(['object']).apply(lambda x: x.astype('category'))
cat_features = [key for key in dict(df_AllNonNumerical.dtypes) if dict(df_AllNonNumerical.dtypes)[key] in ['category']]
df_AllNonNumericalEncoded = pd.get_dummies(df_AllNonNumerical, columns=cat_features, drop_first=True)
df_NonNumericalencoded = df_AllNonNumericalEncoded[:len(df_NonNumericalType.values)]

#Encode Test numerical data
df_TestNonNumericalencoded = df_AllNonNumericalEncoded[len(df_NonNumericalType.values):]

df_NumericalScaled = pd.DataFrame(NumericalScaledData)
df_NumericalScaled.columns = df_NumericalType.columns
df_AllTypes = pd.concat([df_NumericalScaled, df_NonNumericalencoded], axis=1)
df_AllTypes.shape


df_TestNumericalScaled = pd.DataFrame(TestNumericalScaledData)
df_TestNumericalScaled.columns = df_TestNumericalType.columns
df_AllTestTypes = pd.concat([df_TestNumericalScaled, df_TestNonNumericalencoded], axis=1)
df_AllTestTypes.shape


# instantiate the independent variable
from sklearn.tree import *
Y = dataset.iloc[:,-1]


from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
sel2 = VarianceThreshold(threshold=(.5 * (1 - .5)))
varianceThresholdTrain = sel.fit_transform(df_AllTypes.values)
varianceThresholdTest = sel2.fit_transform(df_AllTestTypes.values)

bla = np.delete(varianceThresholdTrain,[10,11,12,13,14],1)
blaTest = np.delete(varianceThresholdTest,[10,11,12,13,14],1)



from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(bla,Y.values, test_size = 0.3)


from sklearn.linear_model import *
lasso = RandomizedLasso(alpha='aic', scaling=0.5, sample_fraction=0.75, n_resampling=200, selection_threshold=0.25, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.2204460492503131e-16, random_state=None, n_jobs=1)
lasso.fit(X_train,Y_train)
lasso.scores_






from cvxopt import *
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
from scipy.optimize import linprog
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler

class DelaunayRegressor(object):
    
    _points = None
    _n_points = None
    _dim = None
    _coef = None
    _scaling = None
    
    def __init__(self, n_points=None, scaling=False):
        self._n_points = n_points
        self._scaling = scaling
    
    def fit(self, X, y):
        y = np.array(y)
        X = np.array(X)
        if self._n_points is None:
            kmeans = AffinityPropagation()
        else:
            kmeans = KMeans(n_clusters=self._n_points)
        if self._scaling:
            reg = LinearRegression()
            reg.fit(X,y)
            self._coef = reg.coef_
            X = X*self._coef
        else:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        kmeans.fit(X)
        self._points = kmeans.cluster_centers_
        self._n_points, self._dim = self._points.shape
        if len(y.shape)==1:
            self._means = np.zeros(self._n_points)
        if len(y.shape)==2:
            self._means = np.zeros((self._n_points, y.shape[1]))
        for j in range(self._n_points):
            self._means[j] = y[kmeans.labels_==j].mean(axis=0)
        
    def score(self, X, y):
        y = np.array(y)
        X = np.array(X)
        self._in_hull = 0
        rss = 0.
        if self._scaling:
            X = X*self._coef
        else:
            X = self._scaler.transform(X)
        y = np.array(y)
        for i in range(len(X)):
            error, weights = self.get_weights_qp(X[i])
            if error == 0:
                self._in_hull += 1
            rss += (weights.dot(self._means)-y[i])**2
        tss = ((y-y.mean())**2).sum()
        return (1-rss/tss)
    
    def predict(self, X):
        X = np.array(X)
        self._in_hull = 0
        if self._scaling:
            X = X*self._coef
        else:
            X = self._scaler.transform(X)
        if len(self._means.shape)==1:
            y_hat = np.zeros(X.shape[0])
        if len(self._means.shape)==2:
            y_hat = np.zeros((X.shape[0], self._means.shape[1]))
        for i in range(len(X)):
            error, weights = self.get_weights_qp(X[i])
            if error == 0:
                self._in_hull += 1
            y_hat[i] = weights.dot(self._means)
        return y_hat
    
    def get_weights_lp(self, point):
        points = self._points
        n_points = self._n_points
        c = np.array([euclidean(points[i], point) for i in range(n_points)])
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[point, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success, lp.x
        
    def get_weights_qp(self, point):
        points = self._points
        n_points, dim = points.shape
        P = matrix(sum(np.outer(points.T[i],points.T[i]) for i in range(dim)))
        q = matrix(sum(-points.T[i]*point[i] for i in range(dim)))
        G = matrix(-np.identity(n_points))
        h = matrix(np.zeros(n_points))
        A = matrix(np.ones((1,n_points)))
        b = matrix(np.ones(1))
        sol = solvers.qp(P, q, G, h, A, b, options={"show_progress":False})
        weights = np.array([sol["x"][i] for i in range(n_points)])
        error = (((self._points.T).dot(weights)-point)**2).sum()
        if error < 1.e-7:
            success, weights_lp = self.get_weights_lp(point)
            if success:
                return 0.0, weights_lp
        return error, weights

reg3 = DelaunayRegressor(scaling = True)
reg3.fit(X_train, Y_train)
reg3.score(X_test, Y_test)
   

#Using Decesion tree regressor to score the split test data
reg = DecisionTreeRegressor(min_samples_leaf=5, max_depth=3,random_state = 0)
reg.fit(X_train, Y_train)
reg.score(X_test, Y_test)

#Using Random forest regressor to score
from sklearn.ensemble import RandomForestRegressor
reg2 = RandomForestRegressor(n_estimators = 500, random_state = 0,oob_score=True)
reg2.fit(X_train,Y_train)
reg2.score(X_test,Y_test),reg2.oob_score_


reg4 = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=20000)
reg4.fit(X_train,Y_train)
reg4.score(X_test,Y_test)          
#ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(X_train, Y_train)          
#none = []
#ElasticNetCV.score(X_test,Y_test)
          
Y1_pred = reg.predict(X_test)
Y2_pred = reg2.predict(X_test) 
Y3_pred = reg3.predict(X_test)
Y4_pred = reg4.predict(X_test)

cv_rmsel=0
rmsel=0
for i in range(len(Y_test)):
    rmsel += (math.log(Y_test[i]+1) - math.log(Y4_pred[i]+1))**2
rmsel=(rmsel/len(Y_test))**0.5
print(rmsel)

cv_rmsel=0
rmsel=0
for i in range(len(Y_test)):
    rmsel += (math.log(Y_test[i]+1) - math.log(Y3_pred[i]+1))**2
rmsel=(rmsel/len(Y_test))**0.5
print(rmsel)


cv_rmsel=0
rmsel=0
for i in range(len(Y_test)):
    rmsel += (math.log(Y_test[i]+1) - math.log(Y1_pred[i]+1))**2
rmsel=(rmsel/len(Y_test))**0.5
print(rmsel)

cv_rmsel=0
rmsel=0
for i in range(len(Y_test)):
    rmsel += (math.log(Y_test[i]+1) - math.log(Y2_pred[i]+1))**2
rmsel=(rmsel/len(Y_test))**0.5
print(rmsel)

#Kaggle competition file upload 

kagReg = RandomForestRegressor(n_estimators = 100, random_state = 0,oob_score=True)
kagReg.fit(varianceThresholdTrain, Y)
predictionVal = kagReg.predict(varianceThresholdTest)
prediction = pd.DataFrame()
prediction = pd.concat([testDataset['Id'],pd.DataFrame(predictionVal)],axis=1)
prediction.columns=["Id","SalePrice"]
#prediction.to_csv('submission250417.csv',index=False)





                   

