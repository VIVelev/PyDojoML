# PyDojoML
A General Purpose Machine Learning Library for Python

## A quick taste of PyDojoML

### How to install
You can easily install it with `pip`.<br>
Copy-paste this in your terminal and run it.
```
pip install pydojoml
```
Good job, now it's time we rock-and-roll!<br>

### Simple Linear Regression example:
```
import numpy as np
from dojo.linear import LinearRegression

# Building the model.
linear_reg = LinearRegression()

# Let's creat some data to fit the model to.
X = np.random.randn(100_000, 100)
y = X @ np.random.rand(100)

# Fitting the model is as easy as a call of a method.
linear_reg.fit(X, y)

# Now lets predict.
prediction = linear_reg.predict(X[:20, :])

```
