import numpy as np

## this is the TRIAL LEVEL kalman filter model

def kalman_filter(X, r, c=1.0, s=1.0, q=0.0):
    """
    X  : (N, D) array of stimulus/feature vectors, one row per trial
    r  : (N,) array of rewards
    c  : prior variance of the weights (prior covariance = c * I)
    s  : reward noise variance (scalar or length-N array)
    q  : transition (diffusion) variance (scalar or length-N array)
    N trials, D stimuli
    """
    X = np.asarray(X, dtype=float)
    r = np.asarray(r, dtype=float).ravel()
    N, D = X.shape

    # initialize weight vector and initial covariance
    w = np.zeros((D, 1))
    C = c * np.eye(D)
    I = np.eye(D)

    # broadcast scalars to length-N arrays
    s = s * np.ones(N)
    q = q * np.ones(N)
    lr = lr * np.ones(N)

    model = []
    for n in range(N):
        # calculate predicted reward
        x_n = X[n, :].reshape(1, D)            # row vector, 1 x D
        rhat = (x_n @ w).item()  

        # calculate prediction error       
        dt = r[n] - rhat     

        # update covariance, calculate                
        C = C + q[n] * I                     # a priori covariance
        P = (x_n @ C @ x_n.T).item() + s[n]    
        K = (C @ x_n.T) / P                    # Kalman gain, D x 1
        w0 = w.copy()


        w = w + K * dt                  

        # update covariance matrix based on data
        C = C - K @ x_n @ C                   

        # model is a list of dictionaries, where model[n] corresponds to the n-th trial
        model.append({"w0": w0.ravel(), "C": C.copy(), "K": K.ravel(),
                      "dt": dt, "rhat": rhat})

    return model