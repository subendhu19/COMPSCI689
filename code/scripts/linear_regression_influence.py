from __future__ import division
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model   
import matplotlib.pyplot as plt

class LinearRegression(object):
    """Linear regression.

    Arguments:
        delta (float): the cut-off point for switching to linear loss
        k (float): parameter controlling the order of the polynomial par of the loss 
    """

    def __init__(self):
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
        
        errors = y - wb.dot(np.insert(X, X.shape[1],1,axis=1).T)
        
        return np.sum(errors**2)/X.shape[0] + 0.1*np.sum(wb**2)

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
        
        xp = np.insert(X, X.shape[1],1,axis=1).T
        errors = y - wb.dot(xp)

        return -2*xp.dot(errors)/X.shape[0] + 2*0.1*wb

    def objective_grad_point(self, wb, X, y):
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
        
        xp = np.insert(X, X.shape[1],1,axis=1).T
        errors = y - wb.dot(xp)

        return -2*xp.dot(errors)/X.shape[0] + 2*0.1*wb/X.shape[0]

    def hessian(self, X, y):
        hessian = np.zeros((2,2))

        for i in range(len(X)):
            xi = X[i].squeeze()
            yi = y[i]
            hessian[0][0] += 2*xi*xi + 0.4
            hessian[0][1] += 2*xi
            hessian[1][0] += 2*xi
            hessian[1][1] += 2 + 0.4

        return hessian/len(X)

    def influences(self, X, y, tX, ty):
        influences = np.zeros(X.shape[0])
        hessian_inv = np.linalg.inv(self.hessian(X,y))
        for i in range(len(influences)):
            xi = np.array([X[i]])
            yi = np.array([y[i]])
            gradtrain = self.objective_grad_point(np.append(self.w, self.b), xi, yi)[:,np.newaxis]
            gradtest = self.objective_grad_point(np.append(self.w, self.b), tX, ty)[:,np.newaxis]
            influences[i] = -1*gradtest.T.dot(hessian_inv).dot(gradtrain)
        return influences

    def influencesloo(self, X, y):
        influences = np.zeros(X.shape[0])
        hessian_inv = np.linalg.inv(self.hessian(X,y))
        for i in range(len(influences)):
            xi = np.array([X[i]])
            yi = np.array([y[i]])
            gradtrain = self.objective_grad_point(np.append(self.w, self.b), xi, yi)[:,np.newaxis]
            influences[i] = -1*gradtrain.T.dot(hessian_inv).dot(gradtrain)
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

    train_X = np.load('../data/q3_train_X.npy')
    train_y = np.load('../data/q3_train_y.npy')

    # 500 cu, 200 cd, 800 fu, 300 fd, 1000 o
    test_X = np.array([train_X[1000]])
    test_y = np.array([train_y[1000]])

    #train_X = train_X[1:,:]
    #train_y = train_y[1:]

    model = LinearRegression()
    model.fit(train_X, train_y)
    #print model.get_params()
    influences =  model.influences(train_X, train_y, test_X, test_y)
    pos_inf_mask = influences > 0
    # print np.max(influences), np.min(influences)
    # print influences[:20]
    errors = np.abs(model.predict(train_X) - train_y)**2

    skLR1 = linear_model.LinearRegression()
    skLR1.fit(train_X, train_y)
    wLR1, bLR1 = skLR1.coef_.squeeze(), skLR1.intercept_
    print wLR1, bLR1
    
    plt.figure(5)
    plt.scatter(train_X, train_y, c=influences)
    #plt.scatter(train_X[pos_inf_mask], train_y[pos_inf_mask], c='blue', s=0.75, label='positive influence')
    #plt.scatter(train_X[~pos_inf_mask], train_y[~pos_inf_mask], c='yellow', s=0.75, label='negative influence')
    plt.colorbar()
    plt.scatter(test_X, test_y, c='red', label="Test point")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Influence of training points on the model')
    vals = np.array(range(-5,7))[:,np.newaxis]
    plt.plot(vals,wLR1*vals + bLR1, color="red", label="Basic linear regression")

    # mask = influences > -5000

    # new_train_X = train_X[mask]
    # new_train_y = train_y[mask]
    # skLR2 = linear_model.LinearRegression()
    # skLR2.fit(new_train_X, new_train_y)
    # wLR2, bLR2 = skLR2.coef_.squeeze(), skLR2.intercept_
    # print wLR2, bLR2
    # vals = np.array(range(-5,7))[:,np.newaxis]
    # plt.plot(vals,wLR2*vals + bLR2, color="green", label="Linear regression removing high influence points")

    plt.legend()

    # plt.figure(2)
    # plt.hist(influences, range(-30000,0,1000))
    # plt.xlabel('influence')
    # plt.ylabel('number of training points')
    # plt.title('Histogram of influences')

    # plt.figure(3)
    # plt.scatter(errors, influences)
    # plt.xlabel("Distance from regression line")
    # plt.ylabel("Influence")
    # plt.title("Influence vs distance from regression line")

    plt.show()

if __name__ == '__main__':
    main()
