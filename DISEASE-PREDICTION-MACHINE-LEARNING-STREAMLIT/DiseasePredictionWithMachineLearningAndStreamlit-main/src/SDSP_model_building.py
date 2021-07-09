import numpy as np
import pandas as pd

data = pd.read_csv('sdsp_patients.csv')
data.replace(" ",np.nan,inplace=True)
data = data.drop(columns = ['Feature_35', 'Feature_36'])
data['Feature_28'].fillna(value='Every Day',inplace=True)
data['Feature_32'].fillna(value='Yes',inplace=True)
data['Feature_33'].fillna(value='No',inplace=True)
data['Feature_47'].fillna(value='No',inplace=True)
data['Feature_48'].fillna(value='No',inplace=True)
data['Feature_49'].fillna(value='No',inplace=True)
data['Feature_50'].fillna(value='No',inplace=True)



data['Feature_3'].fillna(value='127.7', inplace=True)
data['Feature_3'] = data['Feature_3'].astype(float)



def change_values(data):
    for j in range(6,49):
        data.iloc[:, j] = data.iloc[:, j].replace(['Yes', 'No'], [1, 0])


    data.iloc[:, 28] = data.iloc[:, 28].replace(['Every Day', '1-2 Days a Week', '1-2 Days a Month',
                                                       '3-4 Days a Week'], [0, 1, 2, 3])
    data.iloc[:, 29] = data.iloc[:, 29].replace(['Evenings', 'No Difference', 'Mornings'], [0, 1, 2])
    data.iloc[:, 0] = data.iloc[:, 0].replace(['Disease_1', 'Disease_2', 'Disease_3', 'Disease_4'], [1, 2, 3, 4])
    data.iloc[:, 1] = data.iloc[:, 1].replace(['Male', 'Female'], [0, 1])

change_values(data)

data.drop(['Feature_1', 'Feature_5', 'Feature_6', 'Feature_8', 'Feature_10',
                       'Feature_12', 'Feature_38', 'Feature_42', 'Feature_49'], axis=1, inplace=True)
data.drop(data.iloc[:, 8:23], axis=1, inplace=True)
data.drop(data.iloc[:, 9:14], axis=1, inplace=True)
data.drop(data.iloc[:, 14:18], axis=1, inplace=True)


mean = data['Feature_2'].mean()
median = data['Feature_2'].median()
std = data['Feature_2'].std()
max = mean + (2.8 * std)
data.loc[data.Feature_2 > max, 'Feature_2'] = np.nan
data.fillna(median, inplace=True)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit( data[ ["Feature_2","Feature_3","Feature_4","Feature_7","Feature_9","Feature_11","Feature_13",
                "Feature_29","Feature_37","Feature_39","Feature_40","Feature_41","Feature_43","Feature_48",
                "Feature_50"] ] , data['Disease'])


feature_importances = pd.Series(model.feature_importances_, index= data[ ["Feature_2","Feature_3","Feature_4","Feature_7","Feature_9","Feature_11","Feature_13",
                "Feature_29","Feature_37","Feature_39","Feature_40","Feature_41","Feature_43","Feature_48",
                "Feature_50"] ].columns)


array = dict(feature_importances)
dropped_columns = []

for k,v in array.items():
    if v < 0.05:
        dropped_columns.append(k)

data.drop(dropped_columns, axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = data.iloc[:, 1:]
y = data["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train_ND = scaler.fit_transform(X_train)
X_test_ND = scaler.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
y_pred_rfc = rfc.fit(X_train_ND, y_train)

# Saving the model
import pickle
pickle.dump(y_pred_rfc, open('sdsp.pkl', 'wb'))
pickle.dump(data,open("data.pkl","wb"))
