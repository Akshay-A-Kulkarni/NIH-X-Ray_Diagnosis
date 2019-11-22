from x_ray import preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Use the function (provide a path, select input labels, and call the function)
#csv_path = '/Users/gk/Desktop/sample/sample_labels.csv'  #C:\Users\stany\Desktop
csv_path = 'C:/Users/stany/Desktop/project-data-repo/x_ray_1000.csv'
#image_path = '/Users/gk/Desktop/sample/sample/images/'
image_path = 'C:/Users/stany/Desktop/project-data-repo/x_ray_1000_images/'
input_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule',
                 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# input_labels = ['Effusion','Pneumothorax']

#Create an instance of the class
preprocessor = preprocessor(csv_path, image_path, 100, 100)

X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_x_ray_data()

logistic_model = LogisticRegression(solver='lbfgs',random_state=123,max_iter=300)
logistic_model.fit(X_train,y_train)
logistic_pred = logistic_model.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test,logistic_pred)
logistic_accuracy = metrics.accuracy_score(y_test,logistic_pred,normalize=True)
print(logistic_accuracy)
logistic_error = 1 - logistic_accuracy
print(logistic_error)
# logistic_precision = metrics.precision_score(y_test,logistic_pred)
# print(logistic_precision)
# logistic_recall = metrics.recall_score(y_test,logistic_pred)
# print(logistic_recall)
# logistic_f1 = metrics.f1_score(y_test,logistic_pred)
# print(logistic_f1)