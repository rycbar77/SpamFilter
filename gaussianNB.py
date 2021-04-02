import numpy as np
from multinomialNB import MultinomialNB


class GaussianNB(MultinomialNB):
    # calculate mean(mu) and standard deviation(sigma) of the given feature
    def feature_proba(self, feature):
        mu = np.mean(feature)
        sigma = np.std(feature)
        return mu, sigma

    # the probability density for the Gaussian distribution
    def gaussian_proba(self, mu, sigma, x):
        return (1.0 / (sigma * np.sqrt(2 * np.pi)) *
                np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)))

    # given mu and sigma , return Gaussian distribution probability for target_value
    def get_xj_proba(self, mu_sigma, target):
        return self.gaussian_proba(mu_sigma[0], mu_sigma[1], target)
