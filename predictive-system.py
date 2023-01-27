# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:31:06 2023

@author: DELL
"""

import numpy as np
import pickle



# loading the saved model
loaded_model = pickle.load(open('C:/Users/DELL/Downloads/diabetes pred/trained_model.sav', 'rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast Cancer is Benign')
else:
  print('The Breast Cancer is Malignant')
    