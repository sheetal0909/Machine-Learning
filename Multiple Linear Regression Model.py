import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\sheet\Desktop\Full Stack Data Science\Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state=0 )


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor) #regressor is ml model which conside linear regression algorithm
print(regressor.get_params())

y_pred = regressor.predict(X_test) 
print(y_pred)

comparision = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
print(comparision) 

plt.scatter(X_test, y_test, color = 'Red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = (m_slope*12)+c_intercept
print(y_12)

y_15 = (m_slope*15)+c_intercept
print(y_15)

bias = regressor.score(X_train, y_train)
print(bias)

variance = regressor.score(X_test, y_test)
print(variance)


# Statastics For Machine Learning

# Mean
dataset.mean() 
dataset['Salary'].mean()
dataset['YearsExperience'].mean()

# Median
dataset.median() 
dataset['Salary'].median()
dataset['YearsExperience'].median()

# Variance
dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

#Standard Deviation
dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

#Coefficient of Variation
from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])
variation(dataset['YearsExperience'])

#Correlation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])
dataset['Salary'].corr(dataset['Salary'])

# Skewness
dataset.skew()

#Standard Error Measure
dataset.sem()

#Anova -- SSR, SSE, SST
#SSR
y_mean=np.mean(y) 
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

#R2
r_square = 1 - (SSR / SST)
r_square
