import matplotlib.pyplot as plt
import numpy as np

src = "/home/geffen/Downloads/recycle_data_shuffled.npz"

data = np.load(src)
n=1
img = data['x_train'][n]
label = data['y_train'][n]

print(label)
print(img)
plt.show()
print(data['x_train'].shape[0])