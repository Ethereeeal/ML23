import numpy as np

def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc

def softmax(X):
    """ 
    You can take this from handin I
    Compute the softmax of each row of an input matrix (2D numpy array). 
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE
    for i in range(X.shape[0]):
        max = np.amax(X[i,:])
        ln = np.log(np.sum(np.exp(X[i,:] - max), keepdims=True))
        res[i,:] = np.exp(X[i,:]-max-ln)
    ### END CODE
    return res

def relu(x):
    """ Compute the relu activation function on every element of the input
    
        Args:
            x: np.array
        Returns:
            res: np.array same shape as x
        Beware of np.max and look at np.maximum
    """
    ### YOUR CODE HERE
    res = np.maximum(0,x)
    ### END CODE
    return res

def make_dict(W1, b1, W2, b2):
    """ Trivial helper function """
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def get_init_params(input_dim, hidden_size, output_size):
    """ Initializer function using Xavier/he et al Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    Args:
      input_dim: int
      hidden_size: int
      output_size: int
    Returns:
       dict of randomly initialized parameter matrices.
    """
    W1 = np.random.normal(0, np.sqrt(2./(input_dim+hidden_size)), size=(input_dim, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.normal(0, np.sqrt(4./(hidden_size+output_size)), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

  
class NetClassifier():
    
    def __init__(self):
        """ Trivial Init """
        self.params = None
        self.hist = None

    def predict(self, X, params=None):
        """ Compute class prediction for all data points in class X
        
        Args:
            X: np.array shape n, d
            params: dict of params to use (if none use stored params)
        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        pred = None
        ### YOUR CODE HERE
        pred = np.argmax(softmax(relu(X @ params['W1'] + params['b1']) @ params['W2']+ params['b2']),axis=1)
        nn = relu(X@ params['W1'] + params['b1']) @ params['W2'] + params['b2']
        print('The output of the nn is:',nn)
        print('Softmax:',softmax(nn))
        ### END CODE
        return pred
     
    def score(self, X, y, params=None):
        """ Compute accuracy of model on data X with labels y (mean 0-1 loss)
        
        Args:
            X: np.array shape n, d
            y: np.array shape n, 1
            params: dict of params to use (if none use stored params)

        Returns:
            acc: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!
        """
        if params is None:
            params = self.params
        acc = None
        ### YOUR CODE HERE
        acc = np.mean(self.predict(X,params=params) == y)
        ### END CODE
        return acc
    
    @staticmethod
    def cost_grad(X, y, params, c=0.0):
        """ Compute cost and gradient of neural net on data X with labels y using weight decay parameter c
        You should implement a forward pass and store the intermediate results 
        and then implement the backwards pass using the intermediate stored results
        
        Use the derivative for cost as a function for input to softmax as derived above
        
        Args:
            X: np.array shape n, self.input_size
            y: np.array shape n, 1
            params: dict with keys (W1, W2, b1, b2)
            c: float - weight decay parameter
            params: dict of params to use for the computation
        
        Returns 
            cost: scalar - average cross entropy cost with weight decay parameter c
            dict with keys
            d_w1: np.array shape w1.shape, entry d_w1[i, j] = \partial cost/ \partial W1[i, j]
            d_w2: np.array shape w2.shape, entry d_w2[i, j] = \partial cost/ \partial W2[i, j]
            d_b1: np.array shape b1.shape, entry d_b1[1, j] = \partial cost/ \partial b1[1, j]
            d_b2: np.array shape b2.shape, entry d_b2[1, j] = \partial cost/ \partial b2[1, j]
            
        """
        cost = 0
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        d_w1 = None
        d_w2 = None
        d_b1 = None
        d_b2 = None
        labels = one_in_k_encoding(y, W2.shape[1]) # shape n x k
                        
        ### YOUR CODE HERE - FORWARD PASS - compute cost with weight decay and store relevant values for backprop
        # * store the gradient for each data point
        dh_dw1 = 0
        dh_db1 = 0
        dh_dw2 = 0
        dh_db2 = 0
        cost = 0
        #* for each data point
        for i in range(X.shape[0]):
            a = X[i] @ W1
            b = c * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
            #* c is already taken as the weight decay parameter:(
            c_ = a + b1
            d = relu(c_)
            e = d @ W2
            f = e + b2
            g = -np.sum(labels[i] * np.log(softmax(f)))
            h = g + b
            cost += h
        ### END CODE
        
        ### YOUR CODE HERE - BACKWARDS PASS - compute derivatives of all weights and bias, store them in d_w1, d_w2, d_b1, d_b2
            dh_dh = 1
            dh_dg = dh_dh * 1
            # * from the left part
            # * use the derivative derived in the theoritical part
            tmp = np.zeros(f.shape)
            tmp = np.copy(labels[i])
            tmp.reshape((1, labels[i].shape[0]))
            tmp[np.argmax(tmp)] = -1
            dg_df = tmp + softmax(f)

            dh_df = dh_dg * dg_df
            df_de = 1
            dh_de = dh_df * df_de
            de_dd = W2.T
            dh_dd = dh_de @ de_dd

            # * Activations
            tmp = np.copy(c_)
            tmp[tmp <= 0] = 0
            tmp[tmp > 0] = 1
            tmp.tolist()
            # * diagnal!!
            dd_dc = np.diag(tmp[0])
            dh_dc = dh_dd @ dd_dc
            dc_da = 1
            dh_da = dh_dc * dc_da

            # * triky one
            da_dw1 = (X[i].T).reshape((X[i].shape[0], 1))
            d_w1 = da_dw1 @ dh_da
            # * triky one
            de_dw2 = d.T
            d_w2 = de_dw2 @ dh_de

            dc_db1 = 1
            d_b1 = dh_dc * dc_db1
            df_db2 = 1
            d_b2 = dh_df * df_db2

            dh_db = 1
            db_dw1 = 2 * c * W1
            d_w1 += dh_db * db_dw1
            db_dw2 = 2 * c * W2
            d_w2 += dh_db * db_dw2

            dh_dw1 += d_w1
            dh_db1 += d_b1
            dh_dw2 += d_w2
            dh_db2 += d_b2
        ### END CODE
        # the return signature
        return cost/X.shape[0], {'d_w1': dh_dw1/X.shape[0], 'd_w2': dh_dw2/X.shape[0], 'd_b1': dh_db1/X.shape[0], 'd_b2': dh_db2/X.shape[0]}
        
    def fit(self, X_train, y_train, X_val, y_val, init_params, batch_size=32, lr=0.1, c=1e-4, epochs=30):
        """ Run Mini-Batch Gradient Descent on data X, Y to minimize the in sample error for Neural Net classification
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X_train: numpy array shape (n, d) - the training data each row is a data point
           y_train: numpy array shape (n,) int - training target labels numbers in {0, 1,..., k-1}
           X_val: numpy array shape (n, d) - the validation data each row is a data point
           y_val: numpy array shape (n,) int - validation target labels numbers in {0, 1,..., k-1}
           init_params: dict - has initial setting of parameters
           lr: scalar - initial learning rate
           batch_size: scalar - size of mini-batch
           c: scalar - weight decay parameter 
           epochs: scalar - number of iterations through the data to use

        Sets: 
           params: dict with keys {W1, W2, b1, b2} parameters for neural net
        returns
           hist: dict:{keys: train_loss, train_acc, val_loss, val_acc} each an np.array of size epochs of the the given cost after every epoch
           loss is the NLL loss and acc is accuracy
        """
        
        W1 = init_params['W1']
        b1 = init_params['b1']
        W2 = init_params['W2']
        b2 = init_params['b2']
        hist = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        
        ### YOUR CODE HERE
        params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        best_val_cc = self.score(X_val, y_val, params=params)
        self.params = params
        for i in range(epochs):
            # total_loss = 0
            n = X_train.shape[0]
            # * Generate a shuffled sequence
            shuffled_seq = list(np.random.permutation(n))
            # * Use the sequence to shuffle X and Y simultaneously
            shuffled_X = X_train[shuffled_seq]
            shuffled_y = y_train[shuffled_seq]
            # * Mini-batch
            for j in range(0, n, batch_size):
                loss, grad = self.cost_grad(shuffled_X[j: j + batch_size],
                                            shuffled_y[j: j + batch_size],
                                            params=params,
                                            c=c)
                # * Add the loss on the current batch to the total loss
                # total_loss += loss
                # * Update all the parameters
                W1 = W1 - lr * grad['d_w1']
                b1 = b1 - lr * grad['d_b1']
                W2 = W2 - lr * grad['d_w2']
                b2 = b2 - lr * grad['d_b2']
                params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
                # current_loss,_ = self.cost_grad(X_train,y_train,params=params,c=c)
                # print(current_loss)
                # #* if the current parameters make better prediction: store them in self.params
                # current_val_acc = self.score(X_val,y_val,params=params)
                # if current_val_acc > best_val_cc:
                #     self.params = params
                #     best_val_cc = current_val_acc

            # * After one epochs: Record the current performance
            current_train_loss, _ = self.cost_grad(X_train, y_train, params=params, c=c)
            # print('current_train_loss:',current_train_loss)
            current_val_loss, _ = self.cost_grad(X_val, y_val, params=params, c=c)
            current_train_acc = self.score(X_train, y_train, params=params)
            current_val_acc = self.score(X_val, y_val, params=params)
            # print('current_val_acc:',current_val_acc)
            # * Append all the records to hist
            hist['train_loss'].append(current_train_loss)
            hist['train_acc'].append(current_train_acc)
            hist['val_loss'].append(current_val_loss)
            hist['val_acc'].append(current_val_acc)
            # * If the current validation accuracy is better than the best accuracy on the validation set, store the current params
            # * And update the best accuracy
            if current_val_acc > best_val_cc:
                self.params = params
                best_val_cc = current_val_acc

        ### END CODE
        # hist dict should look like this with something different than none
        #hist = {'train_loss': None, 'train_acc': None, 'val_loss': None, 'val_acc': None}
        ## self.params should look like this with something better than none, i.e. the best parameters found.
        # self.params = {'W1': None, 'b1': None, 'W2': None, 'b2': None}
        return hist
        

def numerical_grad_check(f, x, key):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-5
    # d = x.shape[0]
    cost, grad = f(x)
    grad = grad[key]
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:    
        dim = it.multi_index    
        print(dim)
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        # print('cplus cminus', cplus, cminus, cplus-cminus)
        # print('dim, grad, num_grad, grad-num_grad', dim, grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

def test_grad():
    stars = '*'*5
    print(stars, 'Testing  Cost and Gradient Together')
    input_dim = 7
    hidden_size = 1
    output_size = 3
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)

    nc = NetClassifier()
    X = np.random.randn(7, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    f = lambda z: nc.cost_grad(X, y, params, c=1.0)
    print('\n', stars, 'Test Cost and Gradient of b2', stars)
    numerical_grad_check(f, params['b2'], 'd_b2')
    print(stars, 'Test Success', stars)
    
    print('\n', stars, 'Test Cost and Gradient of w2', stars)
    numerical_grad_check(f, params['W2'], 'd_w2')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of b1', stars)
    numerical_grad_check(f, params['b1'], 'd_b1')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of w1', stars)
    numerical_grad_check(f, params['W1'], 'd_w1')
    print('Test Success')

if __name__ == '__main__':
    input_dim = 3
    hidden_size = 5
    output_size = 4
    batch_size = 7
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)
    X = np.random.randn(batch_size, input_dim)
    Y = np.array([0, 1, 2, 0, 1, 2, 0])
    nc.cost_grad(X, Y, params, c=0)
    test_grad()
