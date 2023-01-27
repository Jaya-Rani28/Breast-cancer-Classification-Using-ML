# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:43:33 2023

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/DELL/Downloads/brest cancer prediction dataset/trained_model.sav', 'rb'))

# creating a function for Prediction

def breast_cancer_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The Breast Cancer is Benign'
    else:
      return 'The Breast Cancer is Malignant'
  
    
  
def main():
    
    
    # giving a title
    st.title('Breast Cancer Prediction Web App')
    
    
    # getting the input data from the user
    
    mean_radius = st.text_input('Enter mean radius')
    mean_texture = st.text_input('Enter mean texture')
    mean_perimeter = st.text_input('Enter mean perimeter')
    mean_area = st.text_input('Enter mean area')
    mean_smoothness = st.text_input('Enter mean smoothness')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Breast Cancer Prediction Result'):
        diagnosis = breast_cancer_prediction([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()