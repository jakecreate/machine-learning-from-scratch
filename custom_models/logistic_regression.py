import numpy as np
import math
class LogisticRegression():
    
    def __init__(self):
        self.weight = None # w_0 + w_1 + ... + w_n
        self.cost_list = [] # L
        self.grad_list = [] # gradient norm
        
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def _decision(self, probs, threshold=0.5):
        return np.where(probs >= threshold, 1, 0)
        
    def _cross_entropy_loss(self, y_pred, y_true):
        return - y_true*np.log(y_pred) - (1 - y_true)*np.log(1-y_pred)

    def _cost_log_loss(self, y_pred, y_true):
        return (1/y_pred.shape[0]) * np.sum(self._cross_entropy_loss(y_pred, y_true))
    
    def _add_ones(self, arr):
        one_arr = np.ones(arr.shape[0])
        if arr.ndim == 1:
            return np.vstack([one_arr, arr]).T 
        return np.hstack([one_arr.reshape(-1,1), arr])
        
    def predict(self, X, external=True):
        if external:
            probs = self._sigmoid(self._add_ones(X) @ self.weight)
            decision_pred = self._decision(probs)
            return (decision_pred, probs)
        else:
            return self._sigmoid(X @ self.weight)
    
    def _compute_gradient(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        g_w = np.zeros(n)
        
        y_hat = self.predict(X, external=False)
        g_w[0] = np.sum(y_hat - y)/m
        for j in range(1, n):
            g_w[j] = np.sum((y_hat - y) * X[:, j])/m
        
        return g_w

    def fit(self, X, y, eta=0.01, max_iter=1000):
        
        m = X.shape[0]
        n = X.shape[1]
        
        X_new = self._add_ones(X)
        self.weight = np.random.rand(n + 1)
        
        self._gradient_descent(X_new , y, eta=eta, threshold=0.00001, max_iter=max_iter)
        
    def _gradient_descent(self, X, y, eta=0.001, threshold=100, max_iter=100):
        message = True
        max_delta = 1e+20 # to prevent infinite divergence
        
        iterations = 1
        delta_cost = math.inf
        y_hat = self.predict(X, external=False)
        old_cost = self._cost_log_loss(y_hat, y)

        while delta_cost >= threshold: # threshold is epsilon
            
            iterations += 1
            g_w = self._compute_gradient(X, y)
            self.grad_list.append(np.sqrt(np.sum(g_w**2)))
            self.weight = self.weight - eta*g_w
            
            y_hat = self.predict(X, external=False)
            new_cost = self._cost_log_loss(y_hat, y) 
            
            delta_cost = np.abs(old_cost - new_cost)
            self.cost_list.append(new_cost)
            old_cost = new_cost
            
            if delta_cost > max_delta:
                print(f'# of iterations (divergent): {iterations}')
                message = False
                break
            
        if message:
            print(f'# of iterations: {iterations}')
        
    