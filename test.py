import numpy as np
from utils.lof import LOF
from utils.metrics import euclidean

lof = LOF(2, euclidean)
a = np.array([[0,1], [1,0], [1,1], [69,420]])
print(lof.fit_predict(a))