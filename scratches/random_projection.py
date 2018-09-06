import numpy as np
from sklearn import random_projection
X = np.random.rand(100, 32, 10,10)
transformer = random_projection.GaussianRandomProjection(3)
X_new = transformer.fit_transform(X)
print(X_new.shape)