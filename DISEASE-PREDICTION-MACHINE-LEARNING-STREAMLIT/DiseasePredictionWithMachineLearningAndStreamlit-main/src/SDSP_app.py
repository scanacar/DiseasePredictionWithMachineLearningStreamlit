import streamlit as st
import pandas as pd
import numpy as np
import pickle
import SessionState

session = SessionState.get(run_id=0)

if session is None:
    raise AttributeError(
        "Oh noes. Couldn't get your Streamlit Session object"
        'Are you doing something fancy with threads?')

st.title('I am helping physicians with their diagnosis by using machine learning')
st.header('Please provide as many features as possible below, and click the predict button.')

left, right = st.beta_columns(2)
with left:
    submit = st.button('Predict')
with right:
    clear = st.button('Clear')

if clear:
    session.run_id += 1

st.subheader('Features')

OD = pd.read_csv('sdsp_patients.csv')
OD.replace(" ", np.nan, inplace=True)
OD['Feature_28'].fillna(value='Every Day', inplace=True)
OD['Feature_32'].fillna(value='Yes', inplace=True)
OD['Feature_33'].fillna(value='No', inplace=True)
OD['Feature_47'].fillna(value='No', inplace=True)
OD['Feature_48'].fillna(value='No', inplace=True)
OD['Feature_49'].fillna(value='No', inplace=True)
OD['Feature_50'].fillna(value='No', inplace=True)
OD['Feature_3'] = OD['Feature_3'].astype(float)

data = pickle.load(open('data.pkl', 'rb'))


def user_input_features(OD, data):
    dict2 = {}
    col = list(data.columns)
    for i, j in enumerate(col):
        if i > 0:
            dict2[j] = list(dict(OD[j].value_counts()).keys())
    key = list(dict2.keys())
    val = list(dict2.values())

    inputData = {}
    count1 = 0
    for i in range(len(key)):
        if type(val[i][0]) == type(1.2) or type(val[i][0]) == type(1):
            value1 = st.text_input(key[i], 0, key=count1)
            if value1:
                inputData[key[i]] = value1
            count1 += 1

        else:
            inputData[key[i]] = st.selectbox(key[i], val[i], key=session.run_id)

    dict5 = {}
    for i in range(len(val)):
        temp = 0
        for j in range(len(val[i])):
            dict5[val[i][j]] = temp
            temp += 1
    for i, j in inputData.items():
        for k, l in dict5.items():
            if j == k:
                inputData[i] = l

    features = pd.DataFrame(inputData, index=[0])
    return features


input_df = user_input_features(OD, data)
load_clf = pickle.load(open('sdsp.pkl', 'rb'))

temp = 0
if submit:
    prediction = load_clf.predict(input_df)
    prediction_proba = load_clf.predict_proba(input_df)
    disease_species = np.array([['Disease_1', 0], ['Disease_2', 0], ['Disease_3', 0], ['Disease_4', 0]])
    for i in range(len(disease_species)):
        disease_species[i][1] = list(prediction_proba[0])[i].round(3)

    st.subheader('Predictions')
    st.text('I would suggest you to consider diseases below based on their probabilities attached.')

    count = 1
    for i in sorted(prediction_proba[0], reverse=True):
        for j in range(len(prediction_proba[0])):
            if i.round(3) == float(disease_species[j][1]) and i.round(3) != 0.0 and temp != 3:
                st.write(str(count) + '.', disease_species[j][0],
                         str(round(float(disease_species[j][1]) * 100, 3)) + '%')
                temp += 1
                count += 1
