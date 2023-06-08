import numpy as np


class BayesianOnlineChangePointDetection:
    def __init__(self, hazard, distribution, thresh=1e-5):
        self.hazard = hazard
        self.distribution = distribution
        self.T = 0
        self.beliefs = np.zeros((1, 2))
        self.beliefs[0, 0] = 1.0
        self.threshold = thresh

    def reset_params(self, hazard=None, distribution=None):
        self.T = 0
        self.beliefs = np.zeros((1, 2))
        self.beliefs[0, 0] = 1.0
        # Set new hazard and distribution functions
        self.hazard = hazard or self.hazard
        self.distribution = distribution or self.distribution

    def _expand_belief_matrix(self):
        rows = np.zeros((1, 2))
        self.beliefs = np.concatenate((self.beliefs, rows), axis=0)

    def _shift_belief_matrix(self):
        self.beliefs[:, 0] = self.beliefs[:, 1]
        self.beliefs[:, 1] = 0.0

    def update(self, x):
        self._expand_belief_matrix()

        # Evaluate Predictive Probability (3 in Algortihm 1)
        pi_t = self.distribution.pdf(x)

        # Calculate H(r_{t-1})
        h = self.hazard(self.rt)

        # Calculate Growth Probability (4 in Algorithm 1)
        self.beliefs[1 : self.T + 2, 1] = self.beliefs[: self.T + 1, 0] * pi_t * (1 - h)

        # Calculate Changepoint Probabilities (5 in Algorithm 1)
        self.beliefs[0, 1] = (self.beliefs[: self.T + 1, 0] * pi_t * h).sum()

        # Determine Run length Distribution (7 in Algorithm 1)
        self.beliefs[:, 1] = self.beliefs[:, 1] / self.beliefs[:, 1].sum()

        # Update sufficient statistics (8 in Algorithm 8)
        self.distribution.update_params(x)

        # Update internal state and truncate
        self._shift_belief_matrix()
        self._truncate_beliefs()
        self.T += 1
    
    def _truncate_beliefs(self):
        # Find the index where the cumulative sum of the beliefs exceeds 1-threshold
        og = self.beliefs.shape[0]
        cumsum_beliefs = np.cumsum(self.beliefs[:, 0])
        truncate_index = np.searchsorted(cumsum_beliefs, 1-self.threshold) + 1
        self.beliefs = self.beliefs[:truncate_index]
        self.T -= og - self.beliefs.shape[0]
        if truncate_index < og:
            self.distribution.truncate(truncate_index)

    @property
    def rt(self):
        return np.argmax(self.beliefs[:, 0])

    @property
    def p(self):
        return max(self.beliefs[:, 0])

    @property
    def belief(self):
        return self.beliefs[:, 0]