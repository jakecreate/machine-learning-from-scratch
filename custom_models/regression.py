import numpy as np
import math

class LinearRegression():
    
    def __init__(self):
        self.weight = None # w_0 + w_1 + ... + w_n
        self.cost_list = None # L
        self.grad_list = None # gradient norm
        
    def _cost_mse(self, y_hat, y):
        return np.sum((y_hat - y)**2)/(2*y.shape[0])
    
    def _add_ones(self, arr):
        one_arr = np.ones(arr.shape[0])
        if arr.ndim == 1:
            return np.vstack([one_arr, arr]).T 
        return np.hstack([one_arr.reshape(-1,1), arr])
        
    def fit(self, X, y, eta=0.01, max_iter=1000):
        
        m = X.shape[0]
        n = X.shape[1]
        
        X_new = self._add_ones(X)
        
        w_rand = np.random.rand(n + 1)
        w_rand[0] = 0
        self.weight = w_rand
        
        self._gradient_descent(X_new , y, eta=eta, threshold=0.005, max_iter=max_iter)


    def predict(self, X, external=True):
        if external:
            return self._add_ones(X) @ self.weight
        else:
            return X @ self.weight

    def _compute_gradient(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        
        y_hat = self.predict(X, external=False)
        g_w = np.zeros(n)
        
        g_w[0] = np.sum(y_hat - y)/m
        
        for i in range(1, n):
            g_w[i] = np.sum((y_hat - y) * X[:, i])/m
        
        return g_w
    
    def _gradient_descent(self, X, y, eta, threshold, max_iter):
        delta_cost = math.inf
        y_hat = self.predict(X, external=False)
        old_cost = self._cost_mse(y_hat, y)
        max_delta = 1e+20
        
        self.cost_list = [old_cost]
        self.grad_list = [self._compute_gradient(X, y)]
        
        iterations = 0
        while delta_cost >= threshold:
            
            iterations += 1
            
            g_w = self._compute_gradient(X, y)
            self.grad_list.append(np.sqrt(np.sum(g_w**2)))
            self.weight =self.weight - eta*g_w
            
            y_hat = self.predict(X, external=False)
            new_cost = self._cost_mse(y_hat, y) 
            
            delta_cost = np.abs(old_cost - new_cost)
            self.cost_list.append(new_cost)
            old_cost = new_cost
            
            if delta_cost > max_delta:
                print(f'[error]: gradient descent diverged')
                break
        
        print(f'[complete]: # of iter: {iterations}')