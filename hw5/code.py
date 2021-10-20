import math, numpy as np, matplotlib.pyplot as plt, pdb

seed = 112
np.random.seed(seed)
n = 300     # number of inputs
N = 24      # hidden neurons
eta = 0.001  # learning rate

X = np.random.uniform(0,1,n)        # inputs
V = np.random.uniform(-0.1,0.1,n)
D = np.sin(20*X) + 3*X + V          # desired values

# weights & biases
W1 = np.random.uniform(-0.1,0.1,N) # layer 1
B1 = np.random.uniform(-0.1,0.1,N) # layer 1
W2 = np.random.uniform(-0.5,0.5,N) # layer 2
b2 = np.random.uniform(0,1,1) # layer 2

epochs = 0
losses = []
while(epochs < 100000): # per epoch
    loss = 0
    for i in range(n):
        # forward pass
        y1 = W1 * X[i] + B1
        y2 = np.sum(W2 * np.tanh(y1)) + b2
        
        # calculate loss
        loss += (D[i] - y2)**2
        
        # weight update (backward pass)
        # w <- w - eta * (dE/ dw) # Gradient Descent equation
        W2 = W2 + eta * (D[i] - y2) * np.tanh(y1)
        W1 = W1 + (eta*5) * (D[i] - y2) * (1 - np.tanh(y1)**2) * X[i] * W2 # observe the higher lr used for the initial layer
        B1 = B1 + (eta*5) * (D[i] - y2) * (1 - np.tanh(y1)**2) * W2
        b2 = b2 + eta * (D[i] - y2) * 1

    losses.append(loss/n)
    if epochs % 1000 == 0:
        print(loss/n)
    epochs += 1

epoch_arr = list(range(1, epochs+1))
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.plot(epoch_arr, losses)
plt.title('MSE Loss vs Epochs')
plt.savefig('MSE_Loss_during_BackPropagation_eta_{}.pdf'.format(eta))
plt.show()

output =[]
for i in range(n):
    y1 = W1 * X[i] + B1
    y2 = np.sum(W2 * np.tanh(y1)) + b2
    output.append(y2)

plt.figure()
plt.xlabel('x')
plt.ylabel('output')
plt.axis([0, 1, -2, 5])

plt.scatter(X, D, label='ground truth')
plt.scatter(X, output, label='output')
plt.legend()
plt.title('Curve fit')
plt.savefig('Curve Fit.pdf')
plt.show()