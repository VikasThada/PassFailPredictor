import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
from sklearn.svm import SVC

@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data('../data/PassFail.csv')
st.title('Pass or Fail Prediction')
st.write('Feature 1:Self Study Hours(Daily)')
st.write('Feature 2:Tution Study Hours(Monthly)')


st.sidebar.title("Please Select")
image = Image.open("BinaryCla.png")
st.image(image,use_column_width=True)
st.markdown('<style>body{background-color: yellow;}</style>',unsafe_allow_html=True)


algotype = st.sidebar.selectbox('Select Algorithm Type',('LogisticR','SVM','DecisionTree'))
SS_select = st.sidebar.selectbox('Select Self Study Hours',df['Self_Study_Daily'].unique())
TS_select = st.sidebar.selectbox('Select Tuition Study Hours',df['Tution_Monthly'].unique())

pfd={1:"Pass",0:"Fail"}

# Python main code
x = df.drop('Pass_Or_Fail',axis = 1)
y = df.Pass_Or_Fail

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
lrmodel = LogisticRegression()
lrmodel.fit(x_train,y_train)

dtmodel = DecisionTreeClassifier(max_depth=2)
dtmodel.fit(x_train, y_train)


svmmodel=SVC(kernel='linear')
svmmodel.fit(x_train,y_train)


if algotype=='LogisticR':
    pred=lrmodel.predict([[int(SS_select),int(TS_select)]])
    st.write("Prediction result=",pfd[pred.ravel()[0]])
elif algotype=='SVM':
    pred=svmmodel.predict([[int(SS_select),int(TS_select)]])
    st.write("Prediction result=",pfd[pred.ravel()[0]])
elif algotype=='DecisionTree':
    pred=dtmodel.predict([[int(SS_select),int(TS_select)]])
    st.write("Prediction result=",pfd[pred.ravel()[0]])


        


