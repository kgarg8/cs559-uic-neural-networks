import idx2numpy, numpy as np, matplotlib.pyplot as plt

seed = 250
np.random.seed(seed)

# Load train and test sets
file = 'data/train-images-idx3-ubyte'
train_inputs = idx2numpy.convert_from_file(file)
file = 'data/train-labels-idx1-ubyte'
train_labels = idx2numpy.convert_from_file(file)
file = 'data/t10k-images-idx3-ubyte'
test_inputs = idx2numpy.convert_from_file(file)
file = 'data/t10k-labels-idx1-ubyte'
test_labels = idx2numpy.convert_from_file(file)

W = np.random.uniform(-1, 1, (10, 784)) # initialize weights
n = 60000                               # no. of samples used for training
eta = 0.5                               # learning rate
eps = 0.15                              # threshold

print('seed: {}, #train_samples: {}, lr: {}, threshold: {}'.format(seed, n, eta, eps))

# Training
epoch = 0
errors = []
while(1):
    cur_epoch_errors = 0
    update = 0
    for i in range(n):
        x_i = train_inputs[i].reshape((-1, 1))
        v = W @ x_i
        pred = np.argmax(v)
        if pred != train_labels[i]:
            cur_epoch_errors += 1
        
        d_xi = np.zeros((10,1))     # 10 classes
        d_xi[train_labels[i]] = 1   # One-hot encoding of desired label
        update += eta * (d_xi - np.sign(np.maximum(v, 0))) @ x_i.T
    
    print('Epoch: {}, errors: {}, cur_threshold: {}'.format(epoch, cur_epoch_errors, cur_epoch_errors/ n))

    W += update                     # update weights after the epoch
    errors.append(cur_epoch_errors)
    epoch += 1

    if cur_epoch_errors/ n <= eps:  # threshold condition
        break

print('Train Misclassifications: ', errors)
print('Epochs taken:', epoch)

# Plot epochs vs. misclassifications 
epoch_arr = list(range(1, epoch+1))
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Misclassifications')
plt.plot(epoch_arr, errors)
plt.title('Epochs vs. Number of Misclassifications')
plt.savefig('N_{}_eps_{}_eta_{}_seed_{}.pdf'.format(n, eps, eta, seed))

# Testing
errors = 0
for i in range(test_inputs.shape[0]):
    x_i = test_inputs[i].reshape((-1, 1))
    v = W @ x_i
    pred = np.argmax(v)
    if pred != test_labels[i]:
        errors += 1
print('Test Misclassifications: ', errors)