import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#config
st.set_page_config(
    page_title="Breast Cancer Classification: Logistic Regression",
    page_icon="purple_heart",

)
image = Image.open('bc.jpg')
st.image(image, width=200)


st.write("""
# Breast Cancer Classification
***
""")

st.write("""Classification of Breast Cancer using Logistic Regression.\n
#### Dataset:  \nBreast Cancer Wisconsin (Original) Data Set (UCI Machine Learning Repository)
""")

#sidebar
st.sidebar.header("Input Parameters")
ct = st.sidebar.slider('Clump Thickness', 1, 10, 5, step=1)
ucsize = st.sidebar.slider('Uniformity of cell size', 1, 10, 5, step=1)
ucshape = st.sidebar.slider('Uniformity of cell shape', 1, 10, 5, step=1)
mt = st.sidebar.slider('Marginal Thickness', 1, 10, 5, step=1)
secz = st.sidebar.slider('Single Epithelial Cell Size', 1, 10, 5, step=1)
bn = st.sidebar.slider('Bare Nuclei', 1, 10, 5, step=1)
bc = st.sidebar.slider('Bland Chromatin', 1, 10, 5, step=1)
nn = st.sidebar.slider('Normal Nucleoli', 1, 10, 5, step=1)
mito = st.sidebar.slider('Mitoses', 1, 10, 5, step=1)



def get_user_input():
    param = [ct, ucsize, ucshape, mt, secz, bn, bc, nn, mito] 
    return param

#train model
dataset = pd.read_csv('./breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

classifier = LogisticRegression(random_state=0)
classifier.fit(X, Y)


if st.sidebar.button('Predict'):
    st.session_state.param = get_user_input()
    
    if 'pred' not in st.session_state:
        st.session_state.pred = classifier.predict([st.session_state.param])
    else:
        st.session_state.pred = classifier.predict([st.session_state.param])

st.write("""***""")

if 'pred' in st.session_state:
    # st.write(str(st.session_state.pred[0]))
    userip = {
        'Clump Thickness' : st.session_state.param[0],
        'Uniformity of cell size': st.session_state.param[1],
        'Uniformity of cell shape': st.session_state.param[2],
        'Marginal Thickness': st.session_state.param[3],
        'Single Epithelial Cell Size': st.session_state.param[4],
        'Bare Nuclei': st.session_state.param[5],
        'Bland Chromatin': st.session_state.param[6],
        'Normal Nucleoli': st.session_state.param[7],
        'Mitoses': st.session_state.param[8],
    }
    features = pd.DataFrame.from_dict(userip, orient='index')
    features = features.rename({0: 'Values'}, axis="columns")
    st.write("""
    ### Results :
    #### Input Parameters
    """)
    st.write(features)
    st.write("""
    #### Type of cancer
    """)
    if st.session_state.pred == 2:
        st.write('Benign')
    elif st.session_state.pred == 4:
        st.write('Malignant')  
else:
    st.write('Please set the input parameters.')

