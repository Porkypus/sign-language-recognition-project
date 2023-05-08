from dtw import dtw
from fastdtw import fastdtw
import time
from random import randint
import numpy as np

n = 7500
# Create two sequences
x = [randint(0, 10) for _ in range(n)]
y = [randint(0, 10) for _ in range(n)]

dtw_python_results = []
fastdtw_results = []

for i in range(100):
    st1 = time.time()
    dis1 = dtw(x, y).distance
    ed1 = time.time()

    st2 = time.time()
    dis2 = list(fastdtw(x, y))[0]
    ed2 = time.time()

    dtw_python_results.append(ed1 - st1)
    fastdtw_results.append(ed2 - st2)

# print mean and std of each method
print(
    "dtw-python: mean = {}, std = {}".format(
        np.mean(dtw_python_results), np.std(dtw_python_results)
    )
)
print(
    "fastdtw: mean = {}, std = {}".format(
        np.mean(fastdtw_results), np.std(fastdtw_results)
    )
)
