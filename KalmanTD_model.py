# References:
# Gershman, S.J., 2015. A Unifying Probabilistic View of Associative Learning.
# PLoS Computational Biology 11, 1–20.
# Code from Juliana Sporrer (juliana.sporrer.18@ucl.ac.uk)
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib


def create_stimuli(onset, duration, trial_length):

    stimuli = np.zeros((trial_length), dtype = int)

    if onset >= 0:
        stimuli[onset:(onset + duration)] = 1
        return stimuli

def construct_stim_matrix(stimuli, num_trials):

    stim_matrix = np.matlib.repmat(stimuli, num_trials, 1)

    # stim_matrix = np.zeros((trial_length,num_stimuli*10), dtype = int)

    # start=0
    # finish=1
    # for i in range(num_stimuli):

    #     stim_matrix[:,start*trial_length:finish*trial_length] = np.diag(stimuli[:,i])

    #     start += 1
    #     finish += 1

    return stim_matrix


def Kalman_TD(stim_matrix, reward_matrix, num_stimuli, trial_length, TD, covariance, diffusion_variance, discount_factor, noise_variance, learning_rate):
    '''
    stim_matrix - matrix of stimulus. each row corresponds to a time step
    reward matrix - vector of rewards for each time step
    TD - true or false, indicates if we are using standard TD or kalman TD
    covaraince - initiali covariance of the weights - scalar that is turned into diagonal matrix
    diffusion_variance - aka q, aka process variance, aka tau
    discount_factor - aka gamma, used when weighting long term rewards
    noise_variance - aka measurement noise
    learning_rate - aka alpha
    '''
    
    
    matrix_num_rows = np.size(stim_matrix,0)  # total number of time steps
    
    # create current weights and weights matrix
    weights = np.zeros(num_stimuli)
    weights_matrix = np.zeros((matrix_num_rows, num_stimuli))
    ax_cov = np.zeros(matrix_num_rows)
    x_kalman = np.zeros(matrix_num_rows)

    # turn covariance and diffusion into matrices
    covariance = covariance*np.eye(num_stimuli)
    diffusion_variance = diffusion_variance*np.eye(num_stimuli)

    for i in range(matrix_num_rows-1):
        w_cur = weights_matrix[i, :]

        # h_t 
        h = stim_matrix[i,:] - discount_factor*stim_matrix[i+1,:]

        # reward and prediction error (scalar)
        rhat = np.dot(h, w_cur)
        prediction_error = reward_matrix[i] - rhat

        # Update covariance matrix (matrix)
        covariance = covariance + diffusion_variance

        # Residual covariance (scalar)
        residual_covariance = h @ covariance @ h + noise_variance

        # Kalman gain (row vector)
        kalman_gain = (covariance @ h) / residual_covariance

        # update kalman gain of x and a-x covariance:
        if TD == 0:
                x_kalman[i] = kalman_gain[1]
                ax_cov[i] = covariance[1,0]
        # Update covariance (matrix)
        covariance = covariance - np.outer(kalman_gain, h) @ covariance
    
        # keep track of 
        # standard td
        if TD == 1:
            weights_matrix[i+1, :] = w_cur + learning_rate*h*prediction_error
        # kalman td
        else:
            weights_matrix[i+1, :] = w_cur + kalman_gain*prediction_error

    if TD == 1:
        return weights_matrix
    else: 
        return weights_matrix, x_kalman, ax_cov


# -----------------------------------
# Serial overshadowing Simulation
# -----------------------------------

class serial_overshadowing_simulation():
    """ Simulates the recovery from serial compount overshadowing (with second-order extinction)
            First, creates stimuli.
            Second, creates covariance and reward matrices.
            Third, applies Kalman-TD and standard-TD models.
    """

    def __init__(self, num_stimuli, num_trials, trial_length, onset1, onset2, duration, covariance, noise_variance, diffusion_variance, discount_factor, learning_rate):
        self.num_stimuli = num_stimuli
        self.num_trials = num_trials
        self.trial_length = trial_length
        self.onset1 = onset1
        self.onset2 = onset2
        self.duration = duration
        self.covariance = covariance
        self.noise_variance = noise_variance
        self.diffusion_variance = diffusion_variance
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def initialise_stimuli_ova(self):
        """
        Creates the stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        stimuli_1 = np.zeros((self.trial_length, self.num_stimuli),dtype = int)
        stimuli_1[:,0] = create_stimuli(self.onset1, self.duration, self.trial_length)
        stimuli_1[:,1] = create_stimuli(self.onset2, self.duration, self.trial_length)

        # B -> Y -> +
        stimuli_2 = np.zeros((self.trial_length, self.num_stimuli), dtype = int)
        stimuli_2[:,2] = create_stimuli(self.onset1, self.duration, self.trial_length)
        stimuli_2[:,3] = create_stimuli(self.onset2, self.duration, self.trial_length)

        # A -> -
        stimuli_3 = np.zeros((self.trial_length, self.num_stimuli),dtype = int)
        stimuli_3[:,0] = create_stimuli(self.onset1, self.duration, self.trial_length)


        # Z -> X
        stimuli_4 = np.zeros((self.trial_length, self.num_stimuli),dtype = int)
        stimuli_4[0:2,1] = np.ones(2)
        stimuli_4[2:,4] = np.ones(2)


        return stimuli_1, stimuli_2, stimuli_3, stimuli_4

    def initialise_stimuli_ovb(self):
            """
            Creates the stimulus presented together (or alone) with second-order stimulus extinction.
            """
            # A -> X -> +
            stimuli_1 = np.zeros((self.trial_length, self.num_stimuli),dtype = int)
            stimuli_1[:,0] = create_stimuli(self.onset1, self.duration, self.trial_length)
            stimuli_1[:,1] = create_stimuli(self.onset2, self.duration, self.trial_length)
    
            # B -> Y -> +
            stimuli_2 = np.zeros((self.trial_length, self.num_stimuli), dtype = int)
            stimuli_2[:,2] = create_stimuli(self.onset1, self.duration, self.trial_length)
            stimuli_2[:,3] = create_stimuli(self.onset2, self.duration, self.trial_length)
    
            # B -> -
            stimuli_3 = np.zeros((self.trial_length, self.num_stimuli),dtype = int)
            stimuli_3[:,2] = create_stimuli(self.onset1, self.duration, self.trial_length)
    
    
            # Z -> X
            stimuli_4 = np.zeros((self.trial_length, self.num_stimuli),dtype = int)
            stimuli_4[0:2,1] = np.ones(2)
            stimuli_4[2:,4] = np.ones(2)
    
    
            return stimuli_1, stimuli_2, stimuli_3, stimuli_4

    def initialise_stim_matrix(self, stimuli_1, stimuli_2, stimuli_3, stimuli_4):
        """
        Creates the stimulus matrix for each stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        matrix_stimuli_1 = construct_stim_matrix(stimuli_1, self.num_trials)

        # B -> Y -> +
        matrix_stimuli_2 = construct_stim_matrix(stimuli_2, self.num_trials)

        # A -> -
        matrix_stimuli_3 = construct_stim_matrix(stimuli_3, self.num_trials)

        # Z -> X 
        matrix_stimuli_4 = construct_stim_matrix(stimuli_4, self.num_trials)

        stim_matrix = np.concatenate((matrix_stimuli_1, matrix_stimuli_2, matrix_stimuli_3, matrix_stimuli_4))
        # each row represents the stimulus at a time within a trial
        return stim_matrix

    def initialise_reward_matrix(self):
        """
        Creates the reward matrix for each stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> + and # B -> Y -> +
        reward_stimuli_1 = np.array([0, 0, 0, 1])

        # A -> - and Z -> X
        reward_stimuli_2 = np.zeros((self.trial_length), dtype = int)

        # All the stimuli
        reward_matrix = np.concatenate([np.matlib.repmat(reward_stimuli_1, 1, self.num_trials*2), np.matlib.repmat(reward_stimuli_2, 1, self.num_trials*2)])

        return reward_matrix.flatten()

    def Kalman_TD(self, stim_matrix, reward_matrix):
        """
        Creates the Kalman-TD weight matrix for second-order stimulus extinction simulation.
        """
        weights_matrix_kalmanTD = Kalman_TD(stim_matrix, reward_matrix, self.num_stimuli, self.trial_length, 0, self.covariance, self.diffusion_variance, self.discount_factor, self.noise_variance, self.learning_rate)

        # weights_kalmanTD = weights_matrix_kalmanTD[self.onset2::self.trial_length, (self.trial_length + self.onset2)]
        # weights_kalmanTD = np.concatenate((weights_kalmanTD, weights_matrix_kalmanTD[self.onset2::self.trial_length, ((self.num_stimuli-1)*self.trial_length + self.onset2)]), axis=1)

        return weights_matrix_kalmanTD

    def standard_TD(self, stim_matrix, reward_matrix):
        """
        Creates the standard-TD weight matrix for second-order stimulus extinction simulation.
        """

        weights_matrix_TD = Kalman_TD(stim_matrix, reward_matrix, self.num_stimuli, self.trial_length, 1, self.covariance, self.diffusion_variance, self.discount_factor, self.noise_variance, self.learning_rate)

        return weights_matrix_TD

### Testing the serial overshadowing class
# test = serial_overshadowing_simulation(4, 10, 4, 0, 0, 4, 1, 1, .01, 0.98, 0.3)

# stim = test.initialise_stimuli()
# stim_mat = test.initialise_stim_matrix(stim[0], stim[1], stim[2])

# reward_mat = test.initialise_reward_matrix()

# print("stim mat shape:", stim_mat.shape)

# #print(reward_mat)
# #print("reward_mat shape:", reward_mat.shape)

# ktd_weights = test.Kalman_TD(stim_mat, reward_mat)

# print(ktd_weights)

### Real serial overshadowing set up:
ser_over = serial_overshadowing_simulation(5, 10, 4, 0, 0, 4, 1, 1, .01, 0.98, 0.3)

## OV-A
# intialize stimulus and create matrix
stim = ser_over.initialise_stimuli_ova()
stim_mat = ser_over.initialise_stim_matrix(stim[0], stim[1], stim[2], stim[3])
# print(stim_mat.shape[0])
# print(stim_mat)

# initialize reward vector
reward_mat = ser_over.initialise_reward_matrix()
#print(reward_mat)

## run it for KTD
ktd_weights,_ ,_  = ser_over.Kalman_TD(stim_mat, reward_mat)
time = np.arange(0, reward_mat.shape[0], 1)
# plt.plot(time, ktd_weights[:, 0], label = "A weights")
# plt.plot(time, ktd_weights[:, 1], label = "X weights")
# plt.plot(time, ktd_weights[:, 2], lw = 2, label = "B weights")
# plt.plot(time, ktd_weights[:, 3], label = "Y weights")
# plt.plot(time, ktd_weights[:, 4], label = "Z weights")
# plt.legend()
#plt.show()

## run it for temporal difference
td_weights = ser_over.standard_TD(stim_mat, reward_mat)
print(td_weights)
plt.plot(time, td_weights[:, 0], label = "A weights")
plt.plot(time, td_weights[:, 1], label = "X weights")
plt.plot(time, td_weights[:, 2], lw = 2, label = "B weights")
plt.plot(time, td_weights[:, 3], label = "Y weights")
plt.plot(time, td_weights[:, 4], label = "Z weights")
plt.legend()
plt.show()

# same thing with OV-B
stim_b = ser_over.initialise_stimuli_ovb()
stim_mat_b = ser_over.initialise_stim_matrix(stim_b[0], stim_b[1], stim_b[2], stim_b[3])

ktd_weights_b, _ , _  = ser_over.Kalman_TD(stim_mat_b, reward_mat)
# plt.plot(time, ktd_weights_b[:, 0], label = "A weights")
# plt.plot(time, ktd_weights_b[:, 1], label = "X weights")
# plt.plot(time, ktd_weights_b[:, 2], lw = 2, label = "B weights")
# plt.plot(time, ktd_weights_b[:, 3], label = "Y weights")
# plt.plot(time, ktd_weights_b[:, 4], label = "Z weights")
# plt.legend()
#plt.show()

## run it for temporal difference
td_weights_b = ser_over.standard_TD(stim_mat_b, reward_mat)
plt.plot(time, td_weights_b[:, 0], label = "A weights")
plt.plot(time, td_weights_b[:, 1], label = "X weights")
plt.plot(time, td_weights_b[:, 2], lw = 2, label = "B weights")
plt.plot(time, td_weights_b[:, 3], label = "Y weights")
plt.plot(time, td_weights_b[:, 4], label = "Z weights")
plt.legend()
plt.show()