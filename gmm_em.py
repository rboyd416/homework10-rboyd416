import numpy as np
from scipy.stats import norm

#function which carries out the expectation step of expectation-maximization
def expectation(data, weights, means, varis):
    k = len(means)
    N = len(data)
    gammas = np.zeros((k,N))

    #fill in here
    #code to calculate each gamma = gammas[i][j], the likelihood of datapoint j in gaussian i, from the
    #current weights, means, and varis of the gaussians

    return gammas


#function which carries out the maximization step of expectation-maximization
def maximization(data, gammas):
    k = len(gammas)
    N = len(data)
    weights = np.zeros(k)
    means = np.zeros(k)
    varis = np.zeros(k)

    #fill in here
    #code to calculate each (i) weight = weights[i], the weight of gaussian i, (ii) mean = means[i], the
    #mean of gaussian i, and (iii) var = varis[i], the variance of gaussian i, from the current gammas of the
    #datapoints and gaussians

    return weights, means, varis


#function which trains a GMM with k clusters until expectation-maximization returns a change in log-likelihood of less
#than a tolerance tol
def train(data, k, tol):
    # fill in
    # initializations of gaussian weights, means, and variances according to the specifications
    weights =
    means =
    varis =

    diff = float("inf")
    ll_prev = -float("inf")

    # iterate through expectation and maximization procedures until model convergence
    while(diff >= tol):
        gammas = expectation(data, weights, means, varis)
        weights, means, varis = maximization(data, gammas)
        ll = log_likelihood(data,weights,means,varis)
        diff = abs(ll - ll_prev)
        ll_prev = ll

    return weights, means, varis, ll


#calculate the log likelihood of the current dataset with respect to the current model
def log_likelihood(data, weights, means, varis):
    #fill in

    return ll


def main(datapath, k, tol):
    #read in dataset
    with open(datapath) as f:
        data = f.readlines()
    data = [float(x) for x in data]

    #train mixture model
    weights, means, varis, ll = train(data, k, tol)

    return weights,means,varis,ll
