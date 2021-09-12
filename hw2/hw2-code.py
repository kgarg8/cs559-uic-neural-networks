import numpy as np, matplotlib.pyplot as plt

seed = 112
np.random.seed(seed)
N_values = [100, 1000]

print('****************************seed = {}****************************'.format(seed))
for N in N_values:
    print('*********************N = {}*********************'.format(N))
    
    # (a) Geometric Interpretation of Perceptron

    # Consider following as Optimal Weights
    w0 = np.random.uniform(-0.25,0.25,1)
    w1 = np.random.uniform(-1,1,1)
    w2 = np.random.uniform(-1,1,1)

    # Create sets S (containing 100 vectors), S1 (x.w^T >= 0), S0 (x.W^T < 0)
    S, S1, S0, W = [], [], [], []
    for i in range(N):
        vec_x = np.random.uniform(-1, 1, 2)
        S.append(vec_x.tolist())
        w = w0 + w1*vec_x[0] + w2*vec_x[1]
        W.append(w)
        if w >= 0:
            S1.append(vec_x)
        else:
            S0.append(vec_x)

    # Plot S1
    x1 = [item[0] for item in S1]
    x2 = [item[1] for item in S1]
    plt.figure()
    plt.scatter(x1, x2, marker='o', label='S1')

    # Plot S0
    x1 = [item[0] for item in S0]
    x2 = [item[1] for item in S0]
    plt.scatter(x1, x2, marker='s', label='S0')

    # Plot w_0 + w_1x_1 + w_2x_2 = 0, the separator line
    x = [item[0] for item in S]
    y = [(-w0 - w1*item[0])/w2 for item in S]
    plt.plot(x, y)

    plt.legend()
    plt.title('Geometric Interpretation of Perceptron')
    plt.savefig('Geometric Interpretation of Perceptron_N_{}.pdf'.format(N))

    print('Optimal weights: {} {} {}'.format(w0, w1, w2))
    
    # (b) Perceptron Training algorithm
    
    # Initial Weights - same for all eta values
    w0_i = np.random.uniform(-1,1,1)
    w1_i = np.random.uniform(-1,1,1)
    w2_i = np.random.uniform(-1,1,1)

    eta_values = [1, 10, 0.1]
    for eta in eta_values:
        print('*******Training for eta = {}*******'.format(eta))
        w0_p = w0_i; w1_p = w1_i; w2_p = w2_i
        
        print('Initial weights: {} {} {}'.format(w0_p, w1_p, w2_p))
        
        epochs = 0
        misclassified_arr = []
        
        while(True):
            misclassified = 0
            w0_pp = w0_p; w1_pp = w1_p; w2_pp = w2_p
            for i in range(N):
                w = w0_p + w1_p * S[i][0] + w2_p * S[i][1]
                
                if w>=0 and W[i]<0:                 # Case1 misclassification
                    misclassified += 1
                    w0_pp = w0_pp - eta*1
                    w1_pp = w1_pp - eta*S[i][0]
                    w2_pp = w2_pp - eta*S[i][1]
                
                elif w<0 and W[i]>=0:               # Case2 misclassification
                    misclassified += 1
                    w0_pp = w0_pp + eta*1
                    w1_pp = w1_pp + eta*S[i][0]
                    w2_pp = w2_pp + eta*S[i][1]

            epochs += 1
            w0_p = w0_pp; w1_p = w1_pp; w2_p = w2_pp # Update weights only after the epoch
            misclassified_arr.append(misclassified)
            if misclassified == 0:                   # Convergence condition
                break
        
        print('Final weights: {} {} {}'.format(w0_p, w1_p, w2_p))
        print('Epochs taken:' , epochs)
        
        # Plot epochs vs. misclassifications 
        epoch_arr = list(range(1, epochs+1))
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Misclassifications')
        plt.xticks(epoch_arr)
        plt.plot(epoch_arr, misclassified_arr)
        plt.title('Epochs vs. Number of Misclassifications')
        plt.savefig('eta_{}_N_{}.pdf'.format(eta, N))