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
df.head()


#Check for Correlation
correlation = df['Displacement'].corr(df['Milage'])
correlation


#Plot the loaded data and check for outlier 
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Displacement'], y=df['Milage'])
plt.xlabel("Displacement")
plt.ylabel("Milage")
plt.title("Displacement vs Milage")
plt.show()

#remove ouliers
df = df[~((df['Displacement'] == 3982) & (df['Milage'] == 1449))]
df = df[~((df['Displacement'] == 2987) & (df['Milage'] == 142))]


#Checked again by plotting the data
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Displacement'], y=df['Milage'])
plt.xlabel("Displacement")
plt.ylabel("Milage")
plt.title("Displacement vs Milage")
plt.show()

#Split the data into the test and train
X = df[['Displacement']]
y = df['Milage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build The Model and Check model properties
model = LinearRegression()
model.fit(X_train,y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

#test the model on test data
y_pred = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

#Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")


#Plot The Regression Line
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Displacement'], y=df['Milage'], label='Actual Data')
plt.plot(X, model.predict(X), color='Green', linewidth=2, label='Regression Line')
plt.xlabel('Engine Displacement')
plt.ylabel('Milage (Km\liter)')
plt.title('Linear Regression: Fuel Efficiency vs. Engine Displacement')
plt.legend()
plt.show()

#Predict for own data

new_car = pd.DataFrame({
    'Displacement': [7654]
})

predicted_milage = model.predict(new_car)
predicted_milage
