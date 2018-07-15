import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pandas.read_csv("apartment.csv")

X  = data.drop(['id','price','date','waterfront','lat','long','yr_renovated'],axis=1)
Y = data['price']

X = X.astype('int')
Y = Y.astype('int')

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.5, random_state=2)


models = []
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('LinearRegression', LinearRegression()))
models.append(('GradientBoostingRegressor',ensemble.GradientBoostingRegressor()))
models.append(('BayesianRegressor',linear_model.BayesianRidge()))
# evaluate each model in turn
results = []
names = []

for name, model in models:
    model.fit(X_train, Y_train)
    res = model.score(X_validation, Y_validation)*100

    print(name,':',res,'%')


classifier = LinearRegression()
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_validation)

rms = sqrt(mean_squared_error(Y_validation, predictions))
print('Root Mean Square in Linear Regression: ',rms)

predictions = classifier.predict([[4,2,1200,5500,1,0,5,5,1200,0,1970,98178,1300,5000]])
#predictions = knn.predict([[3,1,1180,5650,1,1,2,3,0,1000,1,1955,2,98178,48,-122.257,1340,5650]])


print('Prediction on given data : ',predictions, "Taka")












