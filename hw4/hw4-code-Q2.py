import math, numpy as np, matplotlib.pyplot as plt

np.random.seed(111)

x = np.array([i+1 for i in range(50)])
u = np.random.uniform(-1, 1, 50)
y = np.array([u_i + i + 1 for i, u_i in enumerate(u)])

N       = len(x)
sum_x   = np.sum(x)
sum_y   = np.sum(y)
sum_x_2 = np.sum([x_i**2 for x_i in x])
sum_x_y = np.sum([x_i*y_i for x_i, y_i in zip(x, y)])

# Linear least squares fit given by y = w1x + w0
w1 = (N*sum_x_y - sum_x*sum_y)/(N*sum_x_2 - sum_x**2)
w0 = (sum_y - w1*sum_x)/N
print(w1, w0)

# Plot
plt.figure()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Linear Least Squares Fit')
plt.scatter(x, y)
plt.plot(x, w0 + w1 * x, 'r')

## Linear least squares fit using Gradient Descent
def gradient_function(w0, w1):
    # Objective function = \sum_{i=1}{50} (y_i - w0 - w1*x_i)**2
    pred = w0 + w1 * x
    g_w0 = 2 * np.sum(pred - y)
    g_w1 = 2 * (pred - y) @ x
    return [g_w0, g_w1]

# initialize parameters
w0_i, w1_i = np.random.uniform(0, 1, 2)
w0_arr, w1_arr, E = [], [], []
iterations = 0; eta = 1e-5
w0 = w0_i; w1 = w1_i

# gradient descent loop
while iterations < 1000:
    g = gradient_function(w0, w1)
    w0 = w0 - eta * g[0]
    w1 = w1 - eta * g[1]
    iterations += 1
    E.append(np.sum((y - w0 - w1*x)**2)) # Energy function

print(w1, w0)

# plot
plt.plot(x, w0 + w1 * x, 'g')
plt.legend(['Closed form solution', 'Gradient Descent', 'Actual Datapoints'])
plt.savefig('least_squares_fit.pdf')