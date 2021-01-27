import numpy as np
import matplotlib.pyplot as plt

from matplotlib import interactive


class NeuralNetwork:
    def __init__(self, inSize, sl2, clsSize, lrt):

        self.iSz=inSize # number of input units\n",
        self.oSz=clsSize  # number of output units\n",
        self.hSz=sl2      # number of hidden units\n",
        # initialize weights\n",
        self.weights1 = (np.random.rand(self.hSz,self.iSz+1)-0.5)/np.sqrt(self.iSz)
        self.weights2 = (np.random.rand(self.oSz,self.hSz+1)-0.5)/np.sqrt(self.hSz)
        self.eta = lrt   # learning rate\n",y

        # Other stuff you think you are going to need\n",

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def feedforward(self, x):
        #print(x)
        # is computed for single training sample\n",
        # z1 = w1^T*x +b1
        x_with_bias = np.array([1])

        for i in range(x.shape[0]):
            x_with_bias = np.vstack((x_with_bias, [x[i]]))

        z1 = np.dot(self.weights1, x_with_bias)

        a1_without_bias = self.sigmoid(z1)
        a1_with_bias = np.array([1])
        for i in range(a1_without_bias.shape[0]):
            a1_with_bias = np.vstack((a1_with_bias, [a1_without_bias[i]]))

        a1 = a1_with_bias
        z2 = np.dot(self.weights2, a1)
        a2 = self.sigmoid(z2)

        # Compute the activation of each neuron j in the hidden layer  𝑎^((𝑙))\n",
        # Until the output unit is reached\n",

        cache = {"z1": z1,
                 "a1": a1,
                 "z2": z2,
                 "a2": a2}

        return a2, cache

    def backprop(self, x, trg, cache):

        # is computed for single training sample\n",

        a1 = cache['a1']
        a2 = cache['a2']
        # Compute the error at the output  𝛿^((𝐿))= trg-𝑎^((3))   --- Here traget-output, not other way around\n",
        dz2 = trg-a2  # (1,1)
        #print(dz2)
        # Compute 𝛿^(2). Look into slides\n",

        d_activation_func = a1 * (1 - a1)  # (5,1)
        dz1 = np.multiply(np.dot(self.weights2.T, dz2),  d_activation_func)  # (5,1) x (1,1) = (5,1)


        # Compute the derivarive of cost finction with respect to \n",
        # each weight in the network  𝜕/(𝜕〖𝜃_𝑖𝑗〗^((𝑙) ) ) 𝐽(𝜃)=〖𝑎_𝑗〗^((𝑙)) 〖𝛿_𝑖〗^((𝑙+1))\n",

        x_with_bias = np.array([1])

        for i in range(x.shape[0]):
            x_with_bias = np.vstack((x_with_bias, [x[i]]))  # (3,1)

        #k = dz1[1:, :]
        dw1 = np.dot(dz1[1:, :], x_with_bias.T)  # (5,1) den (4,1) e indirdik yani dz1 den biasi cikardik, (4,1) x (1,3)

        dw2 = np.dot(dz2, a1.T)  # (1,1) x (1,5) = (1,5)

        return [dw1, dw2]

    def fit(self, X, y, iterNo):

        m = np.shape(X)[1]
        error_function_list = []
        for i in range(iterNo):
            D1 = np.zeros(np.shape(self.weights1))
            D2 = np.zeros(np.shape(self.weights2))

            for j in range(m):
                #ç = np.array([X[:, j]]).T
                [a2, cache] = self.feedforward(np.array([X[:, j]]).T)
                [delta1, delta2] = self.backprop(np.array([X[:,  j]]).T, np.array([y[:, j]]), cache)

                D1 = D1 + delta1
                D2 = D2 + delta2

            self.weights1 = self.weights1 + self.eta*(D1/m)
            self.weights2 = self.weights2 + self.eta*(D2/m)
            # Compute error function after each 100 iterations    \n",

            if i % 100 == 0:
                error_func = 0
                for k in range(m):
                    [a_2, cache] = self.feedforward(np.array([X[:, k]]).T)
                    actual_y = np.array([y[:, k]])
                    error_func += pow(actual_y-a_2, 2)
                error_function_list.append(error_func)

        return error_function_list


    def predict(self,X):

        m = np.shape(X)[1]
        y = np.zeros(m)
        for i in range(m):
            [y[i], empty] = self.feedforward(np.array([X[:, i]]).T)
        return y


X_and = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y_and = np.array([[0, 0, 0, 1]])



#plt.scatter(X_and[0, :], X_and[1, :], c=Y_and, s=40, cmap=plt.cm.Spectral)


#plt.show()
"""
for i in range(1,5):
    ann = NeuralNetwork(X_and.shape[0], (i+1), Y_and.shape[0], 0.05)

    iter_no = 3500
    err_list = ann.fit(X_and, Y_and, iter_no)
    #print(err_list)
    predictions_and = ann.predict(X_and)
    print(i+1)
    print("AND function predictions:")
    print(predictions_and)


    list1 = []
    for i in range(0, iter_no, 100):
        list1.append(i)
    print("Error is:")
    a = err_list[-1]
    print(a)
    plt.scatter(list1, err_list)
    plt.show()
"""

"""
X_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

Y_xor = np.array([[0, 1, 1, 0]])

plt.scatter(X_xor[0, :], X_xor[1, :], c=Y_xor, s=40, cmap=plt.cm.Spectral)


plt.show()

ann = NeuralNetwork(X_xor.shape[0], 2, Y_xor.shape[0], 0.09)

iter_no1 = 30000
err_list1 = ann.fit(X_xor, Y_xor, iter_no1)
predictions_xor = ann.predict(X_xor)
print("XOR function predictions:")
print(predictions_xor)

list11 = []
for i in range(0, iter_no1, 100):
    list11.append(i)

plt.scatter(list11, err_list1)

plt.show()
"""
X_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

Y_xor = np.array([[0, 1, 1, 0]])




for i in range(1,5):
    ann = NeuralNetwork(X_xor.shape[0], (i+1), Y_xor.shape[0], 0.3)

    iter_no = 5000
    err_list = ann.fit(X_xor, Y_xor, iter_no)
    #print(err_list)
    predictions_xor = ann.predict(X_xor)
    print(i+1)
    print("XOR function predictions:")
    print(predictions_xor)
    list1 = []
    for i in range(0, iter_no, 100):
        list1.append(i)
    print("Error is:")
    a = err_list[-1]
    print(a)
    plt.scatter(list1, err_list)
    plt.show()
