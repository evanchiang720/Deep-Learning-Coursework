import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def relu(x):
    return np.maximum(0.0, x)

def derivative_relu(x):
    return (x > 0).astype(float)
    
def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1.0 - np.power(np.tanh(x), 2)

def func(x):
    return sigmoid(x)
    #return relu(x)
    #return tanh(x)

def derivative_func(x):
    return derivative_sigmoid(x)
    #return derivative_relu(x)
    #return derivative_tanh(x)

def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

def loss_function(y, y_pred):
    return np.mean(np.square(y - y_pred))

def loss_curve(loss_values):
    plt.plot(range(100000), loss_values)
    plt.title('Learning curve', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def accuracy_curve(acc_values):
    plt.plot(range(100000), acc_values)
    plt.title('Accuracy curve', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

class Net:
    def __init__(self, x_size, z1_size, z2_size, y_size, n_size):
        self.w1 = np.random.randn(x_size, z1_size)
        self.w2 = np.random.randn(z1_size, z2_size)
        self.w3 = np.random.randn(z2_size, y_size)
        self.b1 = np.random.randn(n_size, z1_size) * 0.1
        self.b2 = np.random.randn(n_size, z2_size) * 0.1
        self.b3 = np.random.randn(n_size, y_size) * 0.1
        self.v1, self.v2, self.v3 = 0, 0, 0
        self.beta = 0 #0.9
        self.kernal_size = 3
        self.conv_out = np.zeros((n_size, z2_size))
        self.conv_w = np.random.randn(self.kernal_size)
        self.conv_b = np.random.randn(1)
        self.delta_conv = np.zeros((n_size, z1_size))

    def forward(self, x):
        self.p1 = np.dot(x, self.w1) + self.b1
        self.z1 = func(self.p1)
        self.p2 = np.dot(self.z1, self.w2) + self.b2
        # Extra: convolutional layer
        #for i in range(self.conv_out.shape[1]):
        #    self.conv_out[:, i] = np.sum(self.z1[:, i:i+self.kernal_size] * self.conv_w, axis=1) + self.conv_b
        self.z2 = func(self.p2)
        #self.z2 = func(self.conv_out)
        self.p3 = np.dot(self.z2, self.w3) + self.b3
        y_pred = sigmoid(self.p3)
        return y_pred

    def backward(self, x, y, y_pred, lr):
        # Get the gradients
        delta_w3 = (y_pred - y) * derivative_sigmoid(y_pred) #d_loss/d_p3 = (d_loss/d_y) * (d_y/d_p3)
        gradient_w3 = np.dot(self.z2.T, delta_w3) / y.shape[0] #d_loss/d_w3 = (d_loss/d_p3) * (d_p3/d_w3)
        delta_w2 = np.dot(delta_w3, self.w3.T) * derivative_func(self.z2) #d_loss/d_p2 = (d_loss/d_p3) * (d_p3/d_z2) * (d_z2/d_p2)
        gradient_w2 = np.dot(self.z1.T, delta_w2) / y.shape[0] #d_loss/d_w2 = (d_loss/d_p2) * (d_p2/d_w2)
        gradient_conv = np.zeros(self.kernal_size)
        #for i in range(self.kernal_size):
        #    gradient_conv[i] = np.sum(delta_w2 * self.z1[:, i:i+self.conv_out.shape[1]]) / y.shape[0]
        
        
        #for i in range(self.conv_out.shape[1]):
        #    self.delta_conv[:, i:i+self.kernal_size] += np.outer(delta_w2[:, i], self.conv_w)
        delta_w1 = np.dot(delta_w2, self.w2.T) * derivative_func(self.z1) #d_loss/d_p1 = (d_loss/d_p2) * (d_p2/d_z1) * (d_z1/d_p1)
        #delta_w1 = self.delta_conv * derivative_func(self.z1)
        gradient_w1 = np.dot(x.T, delta_w1) / y.shape[0] #d_loss/d_w1 = (d_loss/d_p1) * (d_p1/d_w1)

        # Subtract a ratio of the gradients
        self.v3 = self.v3 * self.beta + gradient_w3
        self.v2 = self.v2 * self.beta + gradient_w2
        self.v1 = self.v1 * self.beta + gradient_w1
        self.w3 -= self.v3 * lr
        self.w2 -= self.v2 * lr
        self.w1 -= self.v1 * lr
        self.b3 -= np.mean(delta_w3, axis=0) * lr
        self.b2 -= np.mean(delta_w2, axis=0) * lr
        self.b1 -= np.mean(delta_w1, axis=0) * lr

        self.conv_w -= gradient_conv * lr
        self.conv_b -= np.mean(np.sum(delta_w2, axis=1)) * lr


if __name__ == '__main__':  
    lr = 0.1
    for test in range(2):
        if test == 0:
            X, y = generate_linear(n=100)
            X_test, y_test = generate_linear(n=100)
            nn = Net(2, 3, 3, 1, 100)
        else:
            X, y = generate_XOR_easy()
            X_test, y_test = generate_XOR_easy()
            nn = Net(2, 3, 3, 1, 21)

        # Training
        loss_values = []
        acc_values = []
        for i in range(100000):
            y_pred = nn.forward(X)
            nn.backward(X, y, y_pred, lr)
            y_test_pred = nn.forward(X_test)
            accuracy = np.mean(np.round(y_test_pred).astype(int) == y_test) * 100
            acc_values.append(accuracy)
            loss = loss_function(y, y_pred)
            loss_values.append(loss)
            if i % 5000 == 0:
                print(f"epoch {i} loss : {loss}")
        loss_curve(loss_values)
        
        # Testing
        print()
        y_test_pred = nn.forward(X_test)
        loss = loss_function(y_test, y_test_pred)
        accuracy = np.mean(np.round(y_test_pred).astype(int) == y_test) * 100
        for i in range(y_test.shape[0]):
            print(f"Iter{i:2d} |   Ground truth: {y_test[i][0]} |    prediction: {y_test_pred[i][0]:.5f} | ")
            
        print(f"loss={loss:.5f} accuracy={accuracy:.2f}%\n")
        show_result(X_test, y_test.astype(int), np.round(y_test_pred).astype(int))
