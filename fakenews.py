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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer



data = pandas.read_csv("fakenews.csv")


X = data['text']
Y = data['label']



X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.5, random_state=2)

cv = TfidfVectorizer(min_df=1)


X_train_cv = cv.fit_transform(X_train)
X_validation_cv = cv.transform(X_validation)




models = []
models.append(('MNB', MultinomialNB()))
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('DTC', DecisionTreeClassifier()))
#models.append(('SVC', SVC()))
# evaluate each model in turn
results = []
names = []

seed = 7
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train_cv, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    accuracy = (cv_results.mean())*100
    deviation = cv_results.std()*100
    print(name, '[ Accuracy : ', accuracy , '% Deviation : ', deviation , '% ]')


classifier = MultinomialNB()
classifier.fit(X_train_cv, Y_train)
predictions = classifier.predict(X_validation_cv)
print('Accuracy :',accuracy_score(Y_validation, predictions))
print('Confusion Matrix: \n',confusion_matrix(Y_validation, predictions))
print('Report: \n', classification_report(Y_validation, predictions))


# New data insert
new_data = 'There are two possible explanations. '
new_data_cv = cv.transform([new_data])

predictions = classifier.predict(new_data_cv)
print('Given Data : ',new_data)
print('Prediction on given data : ',predictions)



