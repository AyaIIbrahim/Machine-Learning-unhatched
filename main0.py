import pandas as pd
import sklearn.model_selection as sk
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# 1)Reading the dataset
dataset = pd.read_csv('Salary.csv')
# 2)loading dataset
#   x(features)
X = dataset.iloc[:, :-1].values

#   y(label)
Y = dataset.iloc[:, 1].values

# 3)split data into training and validation/test set
X_train, X_test, Y_train, Y_test = sk.train_test_split(X, Y, test_size=1/3, random_state=0)
# 4)fit model into training
regressor = lm.LinearRegression()
regressor.fit(X_train, Y_train)

# 5)predict the test set/make a prediction
Y_predic = regressor.predict(X_test)

# 6)Ploting
plt.scatter(X_train, Y_train, color = 'gray')
plt.plot(X_train, regressor.predict(X_train), color = 'pink')
plt.title('Salary from Experience')
plt.xlabel('Experience years')
plt.ylabel('Salary')
plt.show()

# New prediction
test_exp = int(input("Enter experience: "))
new_ele = regressor.predict([[test_exp]])
print(f'The predicted salary for a {test_exp} years experience is {new_ele}')