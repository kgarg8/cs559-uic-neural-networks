import math, numpy as np, matplotlib.pyplot as plt

train_loss = [0.003509, 0.000549]
test_loss = [0.277930, 0.276684]
train_acc = [0.9988, 1.00]
test_acc = [0.8740, 1.00]
epoch_arr = [1, 2]

# 1st plot
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epoch_arr, train_loss, label='train_loss')
plt.plot(epoch_arr, test_loss, label='test_loss')
plt.legend()

for x, y in zip(epoch_arr, train_loss):
    label = "{:.6f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points", xytext=(0,10), ha='center')
for x, y in zip(epoch_arr, test_loss):
    label = "{:.6f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Loss vs Epochs')
plt.savefig('Loss vs. Epochs.pdf')
plt.show()

# 2nd plot
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(epoch_arr, train_acc, label='train_acc')
plt.plot(epoch_arr, test_acc, label='test_acc')
plt.legend()

for x, y in zip(epoch_arr, train_acc):
    label = "{:.6f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points", xytext=(0,10), ha='center')
for x, y in zip(epoch_arr, test_acc):
    label = "{:.6f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Accuracy vs Epochs')
plt.savefig('Accuracy vs. Epochs.pdf')
plt.show()

# Logs
# Training Loss: 0.003509, Training Accuracy: 99.88%
# Test Loss: 0.277930, Test Accuracy: 87.40%

# Training Loss: 0.000549, Training Accuracy: 100.00%
# Test Loss: 0.276684, Test Accuracy: 100.00%
