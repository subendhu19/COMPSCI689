from __future__ import division
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

class RobustLinearRegression(object):
    """Generalized robust linear regression.

    Arguments:
        delta (float): the cut-off point for switching to linear loss
        k (float): parameter controlling the order of the polynomial par of the loss 
    """

    def __init__(self):
        self.delta = 1 #cut-off point
        self.k = 1 #polynomial order parameter
        self.w = np.ones(1) #learned coefficients
        self.b = 1 #learned bias
          
    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """
        p = fmin_l_bfgs_b(self.objective, np.append(self.w, self.b), self.objective_grad, args=[X, y])
        self.set_params(p[0][:-1], p[0][-1])
        return self

    def predict(self, X):
        """Predict using the linear model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        return X.dot(self.w) + self.b


    def objective(self, wb, X, y):
        """Compute the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features + 1)):
                training data with an appended columns with all ones.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """
        
        fx = 0
        
        errors = y - wb.dot(np.insert(X, X.shape[1],1,axis=1).T)
        msk = np.absolute(errors) <= self.delta
        fx = fx + np.sum(((errors[msk])**(2*self.k))/(2*self.k))
        fx = fx + np.sum((self.delta**(2*self.k - 1))*(np.absolute(errors[~msk]) - (((2*self.k - 1)/(2*self.k))*self.delta)))
        
        return fx + 0.1*np.sum(wb**2)

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features + 1)):
                training data with an appended columns with all ones.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        
        gx = np.zeros(X.shape[1] + 1)
        
        errors = y - wb.dot(np.insert(X, X.shape[1],1,axis=1).T)
        msk = np.absolute(errors) <= self.delta
        gx = gx + np.sum(((-errors[msk])**(2*self.k - 1)).T*(np.insert(X[msk], X[msk].shape[1],1,axis=1).T),axis=1)
        gx = gx + np.sum(((-errors[~msk])*(self.delta**(2*self.k - 1))/np.absolute(errors[~msk])).T *(np.insert(X[~msk], X[~msk].shape[1],1,axis=1).T),axis=1)
        return gx + 2*0.1*wb

    def hessian(self, wb, X, y):
        error = y - wb.dot(np.insert(X, X.shape[1],1,axis=1).T)
        error = error.squeeze()
        X = X.squeeze()
        if error <= self.delta:
            return np.array([[((2*self.k - 1)*(error**(2*self.k -2))*X*X) + 2*0.1, ((2*self.k - 1)*(error**(2*self.k -2))*X)],[((2*self.k - 1)*(error**(2*self.k -2))*X), ((2*self.k - 1)*(error**(2*self.k -2))) + 2*0.1]])
        else:
            return np.array([[0.2,0],[0,0.2]])

    def influences(self, X, y):
        influences = np.zeros(X.shape[0])
        for i in range(len(influences)):
            xi = np.array([X[i]])
            yi = np.array([y[i]])
            gradtrain = self.objective_grad(np.append(self.w, self.b), xi, yi)[:,np.newaxis]
            influences[i] = -1*gradtrain.T.dot(np.linalg.inv(self.hessian(np.append(self.w, self.b), xi, yi))).dot(gradtrain)
        return influences

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in 
           self.w, self.b.

        Returns: 
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """
        return (self.w, self.b)

    def set_params(self, w, b):
        """Set the parameters of the model. When called, this
           function sets the model parameters that are used
           to make predictions. Assumes parameters are stored in 
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b
        

def main():

    np.random.seed(0)

    #Example code for loading data
    train_X = np.load('../Data/q3_train_X.npy')
    train_y = np.load('../Data/q3_train_y.npy')

    delta = 1
    k = 1
    
    np.random.seed(0)

    train_X = np.load('../data/q3_train_X.npy')
    train_y = np.load('../data/q3_train_y.npy')

    model = RobustLinearRegression()
    model.fit(train_X, train_y)
    #print model.get_params()
    influences =  model.influences(train_X, train_y)
    #print np.max(influences), np.min(influences)
    errors = np.abs(model.predict(train_X) - train_y)**2

    wLR1, bLR1 = model.w.squeeze(), model.b
    #print wLR1, bLR1
    
    plt.figure(1)
    plt.scatter(train_X, train_y, c=influences)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Influence of training points on the model')
    vals = np.array(range(-5,7))[:,np.newaxis]
    plt.plot(vals,wLR1*vals + bLR1, color="red", label="Robust linear regression")

    plt.legend()

    plt.figure(2)
    plt.hist(influences, range(-6,10))
    plt.xlabel('influence')
    plt.ylabel('number of training points')
    plt.title('Histogram of influences')
    
    plt.figure(3)
    plt.scatter(errors, influences, c="blue")
    plt.xlabel("Distance from regression line")
    plt.ylabel("Influence")
    plt.title("Influence vs distance from regression line")

    plt.show()


if __name__ == '__main__':
    main()
