import numpy as np
from scipy.stats import norm
import math
import gmm_visualize

#function which carries out the expectation step of expectation-maximization
def expectation(data, weights, means, varis):
    k = len(means)
    N = len(data)
    gammas = np.zeros((k,N))

    for x in range(k):
      for y in range(N):
        den = 0
        for z in range(k):
          den += weights[z] * norm.pdf(data[y], means[z], math.sqrt(varis[z])) 
        gammas[x][y] = (weights[x] * norm.pdf(data[y], means[x], math.sqrt(varis[x])))/den

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

    for x in range(k):
      weight = 0
      mean = 0
      vari = 0
      for y in range(N):
        weight += gammas[x][y]
        mean += data[y] * gammas[x][y]

      weights[x] = weight/N
      means[x] = mean/weight
      
      for y in range(N):
        vari += gammas[x][y] * ((data[y] - means[x]) ** 2)

      varis[x] = vari/ weight	
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
    weights = [1/k]*k
    means = []
    varis = [1] * k
    
    for x in range(k):
      means.append(min(data) + x * (max(data) - min(data)) / k)
    #print("Means Test: ")
    #print(means)
    diff = float("inf")
    ll_prev = -float("inf")

    # iterate through expectation and maximization procedures until model convergence
    while(diff >= tol):
        #print("Iteration")
        gammas = expectation(data, weights, means, varis)
        #print("Gammas: ")
        #print(gammas)
        weights, means, varis = maximization(data, gammas)
        #print("Weights: ")
        #print(weights)
        #print("Means: ")
        #print(means)
        #print("Varis: ")
        #print(varis)
        ll = log_likelihood(data,weights,means,varis)
        #print("ll: ")
        #print(ll)
        diff = abs(ll - ll_prev)
        ll_prev = ll

    return weights, means, varis, ll


#calculate the log likelihood of the current dataset with respect to the current model
def log_likelihood(data, weights, means, varis):
    
    #fill in
    ll = 0
    for x, val_d in enumerate(data):
      sum = 0
      for y, val_m in enumerate(means):
        sum += weights[y] * norm.pdf(val_d, val_m, math.sqrt(varis[y]))
      ll += np.log(sum)
    return ll


def main(datapath, k, tol):

    #read in dataset
    with open(datapath) as f:
        data = f.readlines()
    data = [float(x) for x in data]
    #data = [12.141406532590782, 4.55489200471575, 2.5673666260367822, 12.19351653969979, 12.78372755227871, 1.2055652005000008, 14.826872112706353, 4.643699755289818, 2.914079156255503, 1.4431893263445528, 14.73938738586368, 4.955765525134837, 13.009633670297589, 4.664401173563705, 4.5191443207949336]
    #train mixture model
    weights, means, varis, ll = train(data, k, tol)
    #gmm_visualize.main(weights, means, varis)
    print("K = " + str(k))
    #print("Log Likelihood = " + str(ll))
    print("Pi = " + str(weights))
    print("Mu = " + str(means))
    print("Varis = " + str(varis))

    return weights,means,varis,ll

if __name__== "__main__":
  main('data.txt',2,1)
  main('data.txt',3,1)
  main('data.txt',4,1)
  main('data.txt',5,1)
  main('data.txt',6,1)

