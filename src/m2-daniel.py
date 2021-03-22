
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.metrics import precision_recall_fscore_support
import pickle
import sys
import os

from sklearn import metrics

from sklearn.metrics import average_precision_score


import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
plt.style.use('ggplot')


def plotROC(n_classes, y_test,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def plotConfusionMatrix(matrix):
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = ['Deceased', 'Hospitalized', 'Nonhospitalized', 
                'Recovered']
    tick_marks = np.arange(len(class_names)) + 0.5
    tick_marks2 = tick_marks
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for XGBoost')
    plt.show()



cases_train = pd.read_csv(str(os.path.abspath(os.path.dirname(__file__))) + r"\data\cases_train_processed.csv")
cases_test = pd.read_csv(str(os.path.abspath(os.path.dirname(__file__))) + r"\data\cases_test_processed.csv")
cases_location = pd.read_csv(str(os.path.abspath(os.path.dirname(__file__))) + r"\data\location_transformed.csv")

cases_trainTrain = cases_train#[:294088]
cases_trainValid = cases_train#[294088:]

dataX = cases_trainTrain
dataY =  cases_trainTrain[['outcome']].copy()
del dataX["outcome"]


lbl = preprocessing.LabelEncoder()
#sex -> unkown: 2 , male: 1, female: 2

dataX['sex'] = lbl.fit_transform(dataX['sex'].astype(str))
dataX['province'] = lbl.fit_transform(dataX['province'].astype(str))
dataX['country'] = lbl.fit_transform(dataX['country'].astype(str))
dataX['date_confirmation'] = lbl.fit_transform(dataX['date_confirmation'].astype(str))
dataX['additional_information'] = lbl.fit_transform(dataX['additional_information'].astype(str))
dataX['source'] = lbl.fit_transform(dataX['source'].astype(str))
dataY['outcome'] = lbl.fit_transform(dataY['outcome'].astype(str))


'''
0 -> Deceased
1 -> Hospitalized
2 -> Nonhospitalized
3 -> Recovered
'''


data_dmatrix = xgb.DMatrix(data=dataX,label=dataY)

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=123)

defaultParams = {"max_depth":6, "learning_rate":0.3, "n_estimators":100,  "objective":'binary:logistic',"booster":'gbtree',"n_jobs":1,"nthread":None,
"gamma":0,"min_child_weight":1,"max_delta_step":0,"subsample":1,"colsample_bytree":1,"colsample_bylevel":1,"reg_alpha":0,"reg_lambda":1,"base_score":0.5,
"random_state":0,"seed":None,"missing":None}

xg_class  = xgb.XGBClassifier(use_label_encoder=False)

xg_class.set_params(**defaultParams)



metLearn=CalibratedClassifierCV(xg_class, method='isotonic', cv=5)
metLearn.fit(X_train, y_train)
'''
#save model
pickle.dump(metLearn, open("../models/xgb_classifier.pkl", 'wb'))

#load model
loaded_model = pickle.load(open("../models/xgb_classifier.pkl", 'rb'))
exit()
'''
trainPredictions = metLearn.predict(X_train)
testPredictions = metLearn.predict(X_test)
c = y_train.values.flatten()
b=y_test.values.flatten()
temp = b
y_score = metLearn.predict_proba(X_test)


lb = preprocessing.LabelBinarizer()
lb.fit(y_test.values.flatten())
hi = lb.transform(y_test.values.flatten())

#plotROC(4,hi,y_score)



test = multilabel_confusion_matrix(c, trainPredictions)
#print(test)

matrix = confusion_matrix(b,testPredictions)
#plotConfusionMatrix(matrix)


accuracyTrain = accuracy_score(y_train, trainPredictions)
accuracyTest = accuracy_score(y_test, testPredictions)
print(testPredictions)
print("Accuracy for train: %.2f%%" % (accuracyTrain * 100.0))
print("Accuracy for validation: %.2f%%" % (accuracyTest* 100.0))

'''
cross fold validation
cv = 2
Accuracy for train: 88.31%
Accuracy for test: 87.93%

cv = 3
Accuracy for train: 88.31%
Accuracy for test: 87.93%

cv = 4
Accuracy for train: 88.43%
Accuracy for test: 88.00%

cv = 5
Accuracy for train: 88.55%
Accuracy for test: 88.12%

cv = 6
Accuracy for train: 88.49%
Accuracy for test: 88.04%
'''
