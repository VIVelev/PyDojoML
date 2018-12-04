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

## Currently supported Machine Learning Models
### Linear Models
  - Linear Regression
  - LASSO
  - Ridge
  - Logistic Regression
  
### Tree Models
  - Classification And Regression Trees (CARTs)
  - Extra-Trees
  
### Support Vector Machines
  - C-SVM
  - Epsilon-SVM
  - Nu-SVMs

### Bayes
  - Naive Bayes algorithm
  
### Ensemble Learning
  - AdaBoost
  - Model Stacking
  
### Clustering
  - Hierarchical Clustering
  - K-Means algorithm
  
### Anomaly detection
  - Univariate and Multivariate Gaussian Distribution
  
### Dimensionality Reduction Techniques
  - Principal Component Analysis
  - Linear Discriminant Analysis
  
### Various metrics
  - classification
  - regression
  - clustering
  
### Model evaluation utils
  - Train-Test splits
  - K-Fold Cross Validation
  
### Data Preprocessing utils
  - encoders
  - normalizers
  - scalers

### Natural Language Processing utils
  - TF-IDF

### Recommender Systems
  - Content Based
  - Collaborative Filtering
