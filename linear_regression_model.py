import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

yash_x = np.array([[1], [2], [3]])
yash_x_train = yash_x
yash_x_test = yash_x
yash_y = np.array([3,2,4])
yash_y_train = yash_y
yash_y_test = yash_y

model = linear_model.LinearRegression()
model.fit(yash_x,yash_y)
yash_y_predicted = model.predict(yash_x_test)

print("Mean squared error is ", mean_squared_error(yash_y_test,yash_y_predicted))
print("Weight: ",model.coef_)
print("Intercept: ",model.intercept_)

plt.scatter(yash_x_test,yash_y_test)
plt.plot(yash_x_test,yash_y_predicted)
plt.show()
