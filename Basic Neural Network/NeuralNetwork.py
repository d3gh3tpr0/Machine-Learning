import numpy as np
import matplotlib.pyplot as plt

data = [[1,1,0], [1,0,1], [0,1,1], [0,0,0]]
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

x_train = data.T[:2, :]
y_train  = data.T[2:, :][0]



def sigmoid(X):
    return 1/(1+ np.e**(-X))
def init_params():
    w1 = np.random.rand(2, 2)
    b1 = -1 * np.random.rand(1, 2)[0]
    w2 = np.random.rand(1, 2)[0]
    b2 = -1* np.random.rand()
    return w1, b1, w2, b2
def forward_propagation(w1, b1, w2, b2, X):
    z1 = w1.dot(X) - b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) - b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def back_propagation(X, w1, b1, w2, b2, z1, a1, z2, a2, Y):
    alpha = 0.1
    e = Y-a2
    phuy_z2 = a2*(1-a2)*e
    delta_w2 = alpha*phuy_z2*a1
    delta_b2 = alpha*(-1)*phuy_z2

    phuy_z1 = a1*(1-a1)*phuy_z2*w2
    delta_w1 = alpha*np.array([phuy_z1]).T*X
    delta_b1 = alpha*(-1)*phuy_z1
    return delta_w1, delta_b1, delta_w2, delta_b2

def update_params(w1, b1, w2, b2, delta_w1, delta_b1, delta_w2, delta_b2):
    w1 = w1 + delta_w1
    b1 = b1 + delta_b1
    w2 = w2 + delta_w2
    b2 = b2 + delta_b2
    return w1, b1, w2, b2

def gradient_decent(x, y, alpha, iteration):
    w1, b1, w2, b2 = init_params()
    
    for j in range(iteration):
         for i in range(len(x.T)):
            #print(x[:, i])
            #print(y[i])
            z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x[:, i])
            d_w1, d_b1, d_w2, d_b2 = back_propagation(x[:, i], w1, b1, w2, b2, z1,
                                                                      a1, z2, a2, y[i])
            w1, b1, w2, b2 = update_params(w1, b1, w2, b2, d_w1, d_b1, d_w2, d_b2)
    return w1, b1, w2, b2
def predict(w1, b1, w2, b2, x):
    z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x)
    return a2

w1, b1, w2, b2 = gradient_decent(x_train, y_train, 0.1, 100000)


                                     



    
