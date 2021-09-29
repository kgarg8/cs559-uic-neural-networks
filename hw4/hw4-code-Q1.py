import math, numpy as np, matplotlib.pyplot as plt

def objective_function(x, y):
    return -math.log(1 - x - y) - math.log(x) - math.log(y)

def gradient_function(x, y):
    return [(1/ (1 - x - y)) - 1/x, (1/ (1 - x - y)) - 1/y]

def hessian_function(x, y):
    return [[1/(x*x) + 1/(1-x-y)**2, 1/(1-x-y)**2], [1/(1-x-y)**2, 1/(1-x-y)**2 + 1/(y*y)]]

x_0, y_0 = 0.19, 0.02 # Initial point
method = 'GD' # or 'Newton'

for eta in [1e-2]:
    x, y, E = [], [], []
    x_i = x_0; y_i = y_0

    iterations = 0
    while iterations < 500:
        g = gradient_function(x_i, y_i)
        
        if method == 'GD':  # Gradient Descent
            delta = g
        else:               # Newton's Method
            H = hessian_function(x_i, y_i)
            H_inv = np.linalg.inv(np.array(H))
            delta = H_inv @ np.array(g)
        
        # update
        x_i = x_i - eta * delta[0]
        y_i = y_i - eta * delta[1]
        
        try:
            E_i = objective_function(x_i, y_i)
        except ValueError:
            E_i = 0
        
        E.append(E_i)
        x.append(x_i)
        y.append(y_i)

        iterations +=1
    
    # Plot Energy function
    epoch_arr = list(range(1, iterations+1))
    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Energies')
    plt.plot(epoch_arr, E)
    plt.title('Energy function progression in {} Method [lr={}]'.format(method, eta))
    plt.savefig('{}_Energy_x_{}_y_{}_eta_{}.pdf'.format(method, x_0, y_0, eta))

    # Plot Trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel('y axis')
    ax.set_xlabel('x axis')
    ax.set_zlabel('energy function')
    plt.title('Trajectory following by points in {} Method [lr={}]'.format(method, eta))
    ax.scatter(x, y, E)
    plt.savefig('{}_Trajectory_x_{}_y_{}_eta_{}.pdf'.format(method, x_0, y_0, eta))