# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```python
Program to implement the linear regression using gradient descent.
Developed by: HARISHA.S
RegisterNumber: 212224230087
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv('/content/50_Startups.csv', header=None)
print(data.head())
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
y1_Scaled = scaler.fit_transform(y)
print('Name: HARISHA S')
print('Register No.:212224230087')
print(X1_Scaled)
theta = linear_regression(X1_Scaled, y1_Scaled)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

## DATA INFORMATION

<img width="695" height="166" alt="image" src="https://github.com/user-attachments/assets/38cb2e0b-fd6c-4f72-af05-d1fb78e77fbf" />

## VALUE OF X

<img width="377" height="781" alt="image" src="https://github.com/user-attachments/assets/d2493a4d-5176-44ab-aaa1-5d8a38239a48" />

<img width="285" height="777" alt="image" src="https://github.com/user-attachments/assets/8ca26833-dc27-4778-bcbb-4a9579918efe" />

## VALUE OF X1 SCALED

<img width="505" height="806" alt="image" src="https://github.com/user-attachments/assets/80405d94-b187-43f6-ac25-bf61d5ad8a38" />

## PREDICTED VALUE

<img width="373" height="25" alt="image" src="https://github.com/user-attachments/assets/b048eb4d-d9ac-451e-9acd-61a772b40d7c" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
