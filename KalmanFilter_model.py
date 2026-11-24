import numpy as np
import matplotlib.pyplot as plt

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


## Okay, now I am going to try and recreate Figure 2. 
sigma_r = 1 # this is fed into s in the model
sigma_w = 1 # this is fed into c in the model
tau     = .01 # this is fed into q in the model

N_pre, N_cond = 10, 10

# Pre group: CS alone, then CS + reward
X_pre = np.ones((N_pre + N_cond, 1))
r_pre = np.concatenate([np.zeros(N_pre), np.ones(N_cond)])
result_pre = kalman_filter(X_pre, r_pre, sigma_r, sigma_w, tau)

# No-Pre group: conditioning only
X_nopre = np.ones((N_cond, 1))
r_nopre = np.ones(N_cond)
result_nopre = kalman_filter(X_nopre, r_nopre, sigma_r, sigma_w, tau)

rhat_pre   = np.array([m["rhat"] for m in result_pre])[N_pre:]   # conditioning phase only
rhat_nopre = np.array([m["rhat"] for m in result_nopre])
gain_pre   = np.array([m["K"][0] for m in result_pre])[:N_pre]   # pre-exposure phase only


plt.plot(list(range(1,11)), rhat_pre, label = "Preconditioned")
plt.plot(list(range(1,11)), rhat_nopre, label = "Not preconditioned")
plt.legend()
#plt.show()

plt.plot(list(range(1,11)), gain_pre)
#plt.show()


### Okay easy. Now I am going to try and recreate Figure 3
# overshadowing - first subplot:
os_state = np.ones((10,2))
os_reward = np.ones(10)

no_os_state = np.concatenate((os_state, np.transpose(np.vstack((np.ones(10), np.zeros(10))))))
no_os_reward = np.concatenate([np.ones(10), np.zeros(10)])

os_result = kalman_filter(os_state, os_reward, sigma_r, sigma_w, tau)
no_os_result = kalman_filter(no_os_state, no_os_reward, sigma_r, sigma_w, tau)

os_weight = os_result[-1]["w0"]
no_os_weight = no_os_result[-1]["w0"]

print("Overshadow weights", os_weight)
print("Unovershadow weights", no_os_weight)


# foward blocking - second subplot:
block1 = np.transpose(np.vstack((np.ones(10), np.zeros(10))))
block2 = np.ones((10,2))
block_state = np.concatenate([block1, block2])
block_reward = np.ones(20)

unblock_state = np.concatenate([block1, block2, block1])
unblock_reward = np.concatenate([np.ones(20), np.zeros(10)])

block_result = kalman_filter(block_state, block_reward, sigma_r, sigma_w, tau)
unblock_result = kalman_filter(unblock_state, unblock_reward, sigma_r, sigma_w, tau)

block_weight = block_result[-1]["w0"]
unblock_weight = unblock_result[-1]["w0"]

print("Blocked weights", block_weight)
print("Unblocked weights", unblock_weight)

# under and over expectation - third subplot
a_stim = np.concatenate([np.ones(10), np.zeros(10), np.ones(10)])
b_stim = np.concatenate([np.zeros(10), np.ones(20)])
over_state = np.transpose(np.vstack([a_stim, b_stim]))
over_reward = np.ones(30)

a_stim2 = np.concatenate([np.ones(10), np.zeros(10), np.ones(20)])
b_stim2 = np.concatenate([np.zeros(10), np.ones(20), np.zeros(10)])
unover_state = np.transpose(np.vstack([a_stim2, b_stim2]))
unover_reward = np.concatenate([np.ones(30), np.zeros(10)])

over_result = kalman_filter(over_state, over_reward, sigma_r, sigma_w, tau)
unover_result = kalman_filter(unover_state, unover_reward, sigma_r, sigma_w, tau)


over_weight = over_result[-1]["w0"]
unover_weight = unover_result[-1]["w0"]

print("Overexpectation weights", over_weight)
print("Unoverexpectation weights", unover_weight)

# conditioned inhibition: fourth subplot
a_stim3 = np.ones(20)
b_stim3 = np.concatenate([np.zeros(10), np.ones(10)])
in_state = np.transpose(np.vstack([a_stim3, b_stim3]))
in_reward = np.concatenate([np.ones(10), np.zeros(10)])

a_stim4 = np.ones(30)
b_stim4 = np.concatenate([np.zeros(10), np.ones(10), np.zeros(10)])
unin_state = np.transpose(np.vstack([a_stim4, b_stim4]))
unin_reward = np.concatenate([np.ones(10), np.zeros(20)])

in_result = kalman_filter(in_state, in_reward, sigma_r, sigma_w, tau)
unin_result = kalman_filter(unin_state, unin_reward, sigma_r, sigma_w, tau)

in_weight = in_result[-1]["w0"]
unin_weight = unin_result[-1]["w0"]

print("Inhibition weights", in_weight)
print("Uninhibition weights", unin_weight)
