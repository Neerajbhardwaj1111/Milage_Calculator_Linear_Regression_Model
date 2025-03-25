#Import Libraries required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load data and check loaded data
df=pd.read_csv("cars_ds_final.csv")


#remove ouliers
df = df[~((df['Displacement'] == 3982) & (df['Milage'] == 1449))]
df = df[~((df['Displacement'] == 2987) & (df['Milage'] == 142))]



#Split the data into the test and train
X = df[['Displacement']]
y = df['Milage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build The Model and Check model properties
model = LinearRegression()
model.fit(X_train,y_train)



#test the model on test data
y_pred = model.predict(X_test)



#Predict for own data


def predict_milage(num):
    new_car = pd.DataFrame({
        'Displacement': [num]
        })

    predicted_milage = model.predict(new_car)
    return predicted_milage

