import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

x = np.array([1,1,1,4,1,3,4,7,6,1,3,2,1,9,0,8,9,8,7,1,1,1,6,2,9,9,9])
plt.figure(figsize=(6,4))
plt.plot(np.arange(len(x)), x)
indices = find_peaks(x, height=None, threshold=None, distance=1,
               prominence=None, width=None, wlen=None, rel_height=None,
               plateau_size=None)
print(indices)
plt.plot(indices[0], x[indices[0]], 'o')
plt.show()

