# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:15:13 2020

@author: EmileVDH

#streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf 
import os
import pickle

##os.chdir(r'C:\Users\EmileVDH\NLPAPP')

model = tf.keras.models.load_model('NLP_Comments_Classification_20200531.h5')
with open('tokenizer_X_20200531.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('tokenizer_Y_20200531.pickle', 'rb') as handle:
    label_tokenizer = pickle.load(handle)

st.title("Comment Moderation Tool ")

st.header('1. Enter Comment')
message=st.text_area('Enter Text Here')
st.button('Submit')

data = [message] 
phrase=list(data)
x = tokenizer.texts_to_sequences(phrase)
paddedx = pad_sequences(x, maxlen=120, padding='post', truncating='post')
probability=model.predict_proba(paddedx).max(axis=1) 
numberpred=("{0:.0%}".format(probability[0]))
       # print(phrase)
if probability==0:
    my_prediction= "No Classification"
elif probability<=.3:
    my_prediction="Accepted"
elif probability>=.7:
    my_prediction="Reject"
else:
    my_prediction="Neutral"
            
#my_prediction = predict(data)
st.header('2. View Result')

st.markdown('Outcome is: ' + str(my_prediction))
st.text('Probability of being Offensive or Spam is: '+  numberpred)

if my_prediction=="Reject" : 
      st.image("Reject.PNG",format='PNG')
    
else :    
      st.image("Approve.PNG",format='PNG')


