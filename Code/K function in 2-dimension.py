import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# random number of points in 2-dimensional space
n = int(100)
x = np.random.rand(n)
y = np.random.rand(n)
# np.random.rand(n) generates n random numbers between 0 and 1
#distance between two points in 2-dimensional space
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# define the distance range for k-function
r = np.linspace(0, 1, 50)

# Find the number of points within a distance r
kt = np.zeros(len(r))
for i in range(len(r)):
    N = 0
    for j in range(len(x)):
        for k in range(len(x)):
            if distance(x[j], y[j], x[k], y[k]) <= r[i]:
                N += 1
    kt[i] = N 

#print the distance range and the number of points within distance r
PointsPerradi = list(zip(r, kt))
print('radious and number of points within distance r', PointsPerradi)

#Show the scatter plot of the random distribution of points and the k-function side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(x, y, s=10, c='b', marker='o', alpha=0.5)
ax1.set_title('Random distribution of points')
ax2.plot(r, kt)
ax2.set_title('k-function')
plt.show()







