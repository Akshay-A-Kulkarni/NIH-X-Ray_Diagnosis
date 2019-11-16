# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:19:42 2019

@author: stany
"""

from x_ray import preprocessor

##Create an instance of the class from Geoff's Preprocessor 
##and develop testing & response datasets split into testing and training subsets.  

#Use the function (provide a path, select input labels, and call the function)
csv_path = 'C:/Users/stany/Desktop/project-data-repo/x_ray_1000.csv'  #C:\Users\stany\Desktop
image_path = 'C:/Users/stany/Desktop/project-data-repo/x_ray_1000/'
input_labels = ['Atelectasis','Cardiomegaly','Consolidation','Effusion','Infiltraion','Mass',
                'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumothorax']

# #Create an instance of the class 
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_x_ray_data(csv_path, image_path, 100, 100, input_labels)

#X_train, X_val, X_test, y_train, y_val, y_test = prepper.prepare_x_ray_data()