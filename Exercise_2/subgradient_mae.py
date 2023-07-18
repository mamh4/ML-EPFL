import numpy as np

def compute_subgradient_mae(y, tx, w): #Notice that earlier we only changed the loss compuation but the gradient was always using the MSE loss function
    """Compute a subgradient of the MAE at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    e = y - tx.dot(w)
    e[np.where(e>0)] = 1
    e[np.where(e<0)] = -1
    e[np.where(e==0)] = 0.3 # or any other value between -1 and 1
    return -tx.T.dot(e)/len(y)