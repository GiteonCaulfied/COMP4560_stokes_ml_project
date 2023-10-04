import h5py
import matplotlib.pyplot as plt
from matplotlib import cm

filename = 'solutions_old/solution_1.h5'
with h5py.File(filename, 'r') as f:
    # Access data within the group
    temperature = f['temperature'][:]
    timestamps = f['timestamps'][:]

print(timestamps)

plt.close(1)
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.imshow(temperature[50, :, :],
          cmap=cm.get_cmap('jet', 10),
          extent=(0, 2, 0, 1))

ax.invert_yaxis()
fig.show()
