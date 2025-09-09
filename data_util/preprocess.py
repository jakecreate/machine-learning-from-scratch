import numpy as np

def train_test_split(X, y, p_train=0.6):
    n = X.shape[0]
    sample_indicies = np.arange(0, n)
    train_mask = np.random.choice(sample_indicies, size=int(p_train * n), replace=False)
    test_mask = sample_indicies[~np.isin(sample_indicies, train_mask)]
    
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]
