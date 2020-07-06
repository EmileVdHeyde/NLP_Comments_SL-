
#Wait for submit to be hit to run the app to show 2 and 3 
#make no select in radio button 
#Only after a selection has been chosen , then save to db



#cd C:\Users\EmileVDH\NLPAPP
#streamlit run app.py """
#pip install streamlit 

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf 
import os
import pickle
import psycopg2
import os
#os.chdir(r'C:\Users\EmileVDH\NLPAPP')

# %% Main Application

model = tf.keras.models.load_model('NLP_Comments_Classification_20200531.h5')
with open('tokenizer_X_20200531.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('tokenizer_Y_20200531.pickle', 'rb') as handle:
    label_tokenizer = pickle.load(handle)

st.title("Comment Moderation Tool ")
st.text('This tool looks for Spam or abusive text comments')

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

#Provide Feedback 
st.header('3. Provide Feedback')
feedback = st.radio( "Is the outcome prediction correct?",('No Response','Agree', 'Disagree'))

a=str(data[0])
b=str(probability[0])
c=str(my_prediction)
d=str(feedback)


# %% Database Section 


##Only run to create the table 
#conn = psycopg2.connect(database="FeedbackDB", user="postgres", password="Evangelos71", host = "127.0.0.1", port = "5432")
conn =psycopg2.connect(database="dfhv0pv3b7q21r", user="yiktglljvlvmvz", password="f56a25f0b75da7ca19be7dd56e87587973c1176bded5e93e34c71b712d712478", host = "ec2-54-161-208-31.compute-1.amazonaws.com", port = "5432")
cur = conn.cursor()
cur.execute('''CREATE TABLE FEEDBACK3
      (ID SERIAL PRIMARY KEY,
      COMMENT           CHAR(1000)   NOT NULL,
      PROB          text     NOT NULL,
      PRED        text ,
      FEEDBACK        text );''')

conn.commit()
conn.close()


#inserts the values into database 
if feedback=='Agree' or feedback=='Disagree' : 
         #conn = psycopg2.connect(database="FeedbackDB", user="postgres", password="Evangelos71", host = "127.0.0.1", port = "5432")
         conn =psycopg2.connect(database="dfhv0pv3b7q21r", user="yiktglljvlvmvz", password="f56a25f0b75da7ca19be7dd56e87587973c1176bded5e93e34c71b712d712478", host = "ec2-54-161-208-31.compute-1.amazonaws.com", port = "5432")
         cur = conn.cursor()
         cur.execute("INSERT INTO FEEDBACK3 (COMMENT,PROB,PRED,FEEDBACK) VALUES(%s, %s, %s, %s)", ( a, b, c, d))
         conn.commit()
         conn.close()
         st.header('Thanks for you feedback!!')
else :
        ''



