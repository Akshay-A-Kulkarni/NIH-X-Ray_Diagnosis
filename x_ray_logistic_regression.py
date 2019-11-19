from x_ray import preprocessor
import numpy as np
from pathlib import Path
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

# Use the function (provide a path, select input labels, and call the function)

# input_labels_1000 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Infiltration',
#                      'Mass', 'No Finding', 'Nodule','Pleural_Thickening', 'Pneumothorax']
#
# csv_path = '/Users/gk/Desktop/sample/Data_Entry_2017.csv'
# x_ray_dev = pd.read_csv(csv_path, header = 0).drop('Unnamed: 11',axis = 1)
# x_ray_dev = x_ray_dev[~x_ray_dev['Finding Labels'].str.contains('\\|')]

# x_ray_1000 = pd.DataFrame()
#
# for i in input_labels_1000:
#     label = x_ray_dev[x_ray_dev['Finding Labels'] == i].sample(n=1000)
#     x_ray_1000 = x_ray_1000.append(label)#.sample(frac=1)


# x_ray_1000.to_csv('/Users/gk/Desktop/sample/x_ray_1000.csv',index=False)

# x_ray_1000 = x_ray_1000.sort_values('Image Index')

# print("label counts: \n",x_ray_1000['Finding Labels'].value_counts())

# images = []
# for i in x_ray_1000['Image Index']:
#     images.append(i)
# images = set(images)
#

# sample_path = '/Users/gk/Desktop/sample/Data_Entry_2017.csv'
# sample_df = pd.read_csv(sample_path, header = 0).drop('Unnamed: 11',axis = 1)

# import os
#
# x_ray_1000_list = []
# x_ray_1000_new = pd.DataFrame()
#
# source = os.listdir("/Users/gk/Desktop/sample/x_ray_1000_images")
# source_set = set(source)
#
# for image in sample_df['Image Index']:
#     if image in source_set:
#         x_ray_1000_new = x_ray_1000_new.append(sample_df.loc[sample_df['Image Index'] == image])
#
# x_ray_1000_new.to_csv('/Users/gk/Desktop/sample/x_ray_1000_new.csv',index=False)


# import shutil
# for file in source:
#     if file in images:
#         count = count + 1
#         # shutil.copyfile("/Users/gk/Downloads/x_ray_images/"+file,
#         #                 "/Users/gk/Downloads/x_ray_1000_images/"+file)
#


# test = set(images) & set(source)
# print(len(test))

# print("label counts: \n",x_ray_dev['Finding Labels'].value_counts())






#
# x_ray_dev = pd.read_csv(csv_path, header=0)
# print(x_ray_dev)


# input_labels_1000 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Infiltration',
#                      'Mass', 'No Finding', 'Nodule','Pleural_Thickening', 'Pneumothorax']


# #Create an instance of the class

#
# preprocessor = preprocessor(csv_path, image_path, 100, 100,labels=input_labels)
# X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_x_ray_data()
#
# logistic_model = LogisticRegression(solver = 'lbfgs',multi_class='auto',max_iter=1000)
# logistic_model.fit(X_train,y_train)
# logistic_pred = logistic_model.predict(X_test)
#
# logistic_accuracy = accuracy_score(y_test,logistic_pred)
# print(logistic_accuracy)
# logistic_error = 1 - logistic_accuracy
# print(logistic_error)
# logistic_precision = sklearn.metrics.precision_score(y_test,logistic_pred,average=None)
# print(logistic_precision)
# logistic_recall = sklearn.metrics.recall_score(y_test,logistic_pred,average=None)
# print(logistic_recall)
# logistic_f1 = sklearn.metrics.f1_score(y_test,logistic_pred,average=None)
# print(logistic_f1)

# conf_matrix = sklearn.metrics.confusion_matrix(y_test,logistic_pred)
# print(conf_matrix)

# input_labels = ['Atelectasis','Cardiomegaly','Consolidation','Effusion','Infiltration','Mass',
#                'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumothorax']

input_labels = ['Atelectasis','Cardiomegaly','Consolidation','Effusion','Infiltration','Mass',
                'Nodule', 'Pleural_Thickening', 'Pneumothorax']

csv_path = '/Users/gk/Desktop/sample/x_ray_1000.csv'
image_path = '/Users/gk/Desktop/sample/x_ray_1000_images/'

preprocessor = preprocessor(csv_path, image_path, 100, 100, input_labels)

X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_x_ray_data()

logistic_model = LogisticRegression(solver = 'lbfgs',multi_class='multinomial',max_iter=2000)
logistic_model.fit(X_train,y_train)
logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test,logistic_pred)
print(logistic_accuracy)

logistic_precision = sklearn.metrics.precision_score(y_test,logistic_pred,average='macro')
print(logistic_precision)
logistic_recall = sklearn.metrics.recall_score(y_test,logistic_pred,average='macro')
print(logistic_recall)
logistic_f1 = sklearn.metrics.f1_score(y_test,logistic_pred,average='macro')
print(logistic_f1)


# penalty = ['l1','l2']

# C = np.logspace(0,4,10)
# max_iter = [100,200,300,400,500]
#solver = ['sag', 'saga','lbfgs']

# hyperparameters = dict(max_iter = max_iter)
#
# clf = GridSearchCV(logistic_model, hyperparameters, cv=5, verbose=0)
# best_model = clf.fit(X_train,y_train)

# print('Best Penalty: ', best_model.best_estimator_.get_params()['penalty'])
# print('Best C: ', best_model.best_estimator_.get_params()['C'])
# print('Best Solver: ', best_model.best_estimator_.get_params()['solver'])
# print('Best Max_Iter: ', best_model.best_estimator_.get_params()['max_iter'])

# conf_matrix = sklearn.metrics.confusion_matrix(y_test,logistic_pred)
# logistic_accuracy = sklearn.metrics.accuracy_score(y_test,logistic_pred,normalize=True)
# print(logistic_accuracy)


