import numpy as np
import math

EMA = [[100,100, math.nan], [100,100, math.nan]]

print(np.array(EMA)[:,0], np.diff(np.array(EMA)[:,0], axis=0))
print(np.array(EMA)[:,0], np.diff((np.array(EMA)[:,0]), axis=0), np.sum(np.diff((np.array(EMA)[:,0]), axis=0), axis=0))

MAL = np.sum(np.diff(np.array(EMA), axis=0), axis=0) / np.sum(np.abs(np.diff(np.array(EMA), axis=1)), axis=0)
