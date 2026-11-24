# References:
# Gershman, S.J., 2015. A Unifying Probabilistic View of Associative Learning.
# PLoS Computational Biology 11, 1–20.
# Code from Juliana Sporrer (juliana.sporrer.18@ucl.ac.uk)
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def create_stimuli(onset, duration, trial_length):

    stimuli = np.zeros((trial_length), dtype = np.int)

    if onset >= 0:
        stimuli[onset:(onset + duration)] = 1
        return stimuli

def construct_stim_matrix(stimuli, trial_length, num_stimuli):

    stim_matrix = np.zeros((trial_length,num_stimuli*10), dtype = np.int)

    start=0
    finish=1
    for i in range(num_stimuli):

        stim_matrix[:,start*trial_length:finish*trial_length] = np.diag(stimuli[:,i])

        start += 1
        finish += 1

    return stim_matrix


def Kalman_TD(stim_matrix, reward_matrix, num_stimuli, trial_length, TD, covariance, diffusion_variance, discount_factor, noise_variance, learning_rate):
    matrix_num_columns = np.size(stim_matrix,1)
    matrix_num_rows = np.size(stim_matrix,0)

    stim_matrix = np.asmatrix(np.concatenate((stim_matrix, np.zeros( (1,(num_stimuli * trial_length)))), axis=0)) #adds buffer

    # create current weights and weights matrix
    weights = np.asmatrix(np.zeros((matrix_num_columns,1)))
    weights_matrix = np.asmatrix(np.zeros((matrix_num_rows,matrix_num_columns)))

    # turn covariance and diffusion into matrices
    covariance = covariance*np.eye(matrix_num_columns)
    diffusion_variance = diffusion_variance*np.eye(matrix_num_columns)

    for i in range(matrix_num_rows):

        values = stim_matrix[i,:]*weights
        
        weights_matrix[i,::] = weights.flatten()

        # h_t 
        temporal_difference = stim_matrix[i,:] - discount_factor*stim_matrix[i+1,:]

        # immediate reward
        rhat = temporal_difference*weights
        prediction_error = reward_matrix[0,i] - rhat

        # update covariance matrix: account for potential drift of weights
        covariance = covariance + diffusion_variance

        # kalman gain: nx1 vector
        residual_covariance = temporal_difference*covariance*temporal_difference.transpose() + noise_variance
        kalman_gain = covariance*temporal_difference.transpose() / residual_covariance

        # standard td
        if TD == 1:
            weights = weights + learning_rate*temporal_difference.transpose()*prediction_error
        # kalman td
        else:
            weights = weights + kalman_gain*prediction_error

        # update covariance matrix
        covariance = covariance - kalman_gain*temporal_difference*covariance

    return weights_matrix


## BLAKE note: I'm not sure what is going on below this line - I took this file from a repo built by Juliana Sporrer
## and changed some of the notation to make it more clear. Everything above this has all been checked: below this line is 
## the acutal simulation of the KTD model
# -----------------------------------
# Serial overshadowing Simulation
# -----------------------------------

class serial_overshadowing_simulation():
    """ Simulates the recovery from serial compount overshadowing (with second-order extinction)
            First, creates stimuli.
            Second, creates covariance and reward matrices.
            Third, applies Kalman-TD and standard-TD models.
    """

    def __init__(self, num_stimuli, num_trials, trial_length, onset1, onset2, duration, covariance, noise_variance, diffusion_variance, discount_factor, trace_decay, learning_rate):
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
        self.trace_decay = trace_decay
        self.learning_rate = learning_rate

    def initialise_stimuli(self):
        """
        Creates the stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        stimuli_1 = np.zeros((self.trial_length, self.num_stimuli),dtype = np.int)
        stimuli_1[:,0] = create_stimuli(self.onset1, self.duration, self.trial_length)
        stimuli_1[:,1] = create_stimuli(self.onset2, self.duration, self.trial_length)

        # B -> Y -> +
        stimuli_2 = np.zeros((self.trial_length, self.num_stimuli), dtype = np.int)
        stimuli_2[:,2] = create_stimuli(self.onset1, self.duration, self.trial_length)
        stimuli_2[:,3] = create_stimuli(self.onset2, self.duration, self.trial_length)

        # A -> -
        stimuli_3 = np.zeros((self.trial_length, self.num_stimuli),dtype = np.int)
        stimuli_3[:,0] = create_stimuli(self.onset1, self.duration, self.trial_length)

        return stimuli_1, stimuli_2, stimuli_3

    def initialise_stim_matrix(self, stimuli_1, stimuli_2, stimuli_3):
        """
        Creates the stimulus matrix for each stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        matrix_stimuli_1 = construct_stim_matrix(stimuli_1, self.trial_length, self.num_stimuli)

        # B -> Y -> +
        matrix_stimuli_2 = construct_stim_matrix(stimuli_2, self.trial_length, self.num_stimuli)

        # A -> -
        matrix_stimuli_3 = construct_stim_matrix(stimuli_3, self.trial_length, self.num_stimuli)

        # All the stimuli
        stim_matrix = np.concatenate((np.matlib.repmat( (matrix_stimuli_1), self.num_trials, 1), np.matlib.repmat( (matrix_stimuli_2), self.num_trials, 1)))
        stim_matrix = np.concatenate((stim_matrix, np.matlib.repmat( (matrix_stimuli_3), self.num_trials, 1)))

        return stim_matrix

    def initialise_reward_matrix(self, stimuli_1, stimuli_2, stimuli_3):
        """
        Creates the reward matrix for each stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        reward_stimuli_1 = create_stimuli((self.onset2 + self.duration - 1), 1, self.trial_length)

        # B -> Y -> +
        reward_stimuli_2 = create_stimuli((self.onset2 + self.duration - 1), 1, self.trial_length)

        # A -> -
        reward_stimuli_3 = np.zeros((self.trial_length), dtype = np.int)

        # All the stimuli
        reward_matrix = np.asmatrix( np.append( np.matlib.repmat( (reward_stimuli_1, reward_stimuli_2), self.num_trials, 1), np.matlib.repmat(reward_stimuli_3, self.num_trials, 1)))

        return reward_matrix

    def Kalman_TD(self, stim_matrix, reward_matrix):
        """
        Creates the Kalman-TD weight matrix for second-order stimulus extinction simulation.
        """
        weights_matrix_kalmanTD = Kalman_TD(stim_matrix, reward_matrix, self.num_stimuli, self.trial_length, 0, self.covariance, self.diffusion_variance, self.discount_factor, self.noise_variance, self.learning_rate)

        weights_kalmanTD = weights_matrix_kalmanTD[self.onset2::self.trial_length, (self.trial_length + self.onset2)]
        weights_kalmanTD = np.concatenate((weights_kalmanTD, weights_matrix_kalmanTD[self.onset2::self.trial_length, ((self.num_stimuli-1)*self.trial_length + self.onset2)]), axis=1)

        return weights_kalmanTD

    def standard_TD(self, stim_matrix, reward_matrix):
        """
        Creates the standard-TD weight matrix for second-order stimulus extinction simulation.
        """

        weights_matrix_TD = Kalman_TD(stim_matrix, reward_matrix, self.num_stimuli, self.trial_length, 1, self.covariance, self.diffusion_variance, self.discount_factor, self.noise_variance, self.learning_rate)

        weights_TD = weights_matrix_TD[self.onset2::self.trial_length, (self.trial_length + self.onset2)]
        weights_TD = np.concatenate((weights_TD, weights_matrix_TD[self.onset2::self.trial_length, ((self.num_stimuli-1)*self.trial_length + self.onset2)]), axis=1)

        return weights_TD
