import monte_carlo_pi
import matplotlib
from matplotlib import pyplot as plt
print("Enter the desired number of trials for π estimation:")
num_trials = input()
num_trials = int(num_trials)
# here, we'll run the monte carlo sim as many times as specified by the user, 
# and store the results in the dictionary "estimates"
estimates = []
for n in range(num_trials):
    pi_estimate, x, y, inside = monte_carlo_pi.estimate_pi(10000, seed=None)
    estimates.append(pi_estimate)
    #print(estimates[n])
# Now we can analyze through quite a few statistical methods, we'll intuitively start first with mean
# then move on to variance, standard deviation, and finally confidence intervals.
mean_estimate = sum(estimates) / len(estimates)
print(f"Mean of π estimates after {num_trials} trials: {mean_estimate}")
squared_diffs = [(est - mean_estimate) ** 2 for est in estimates]
standard_deviation = (sum(squared_diffs) / len(estimates)) ** 0.5
print(standard_deviation)
#test
plt.scatter(range(num_trials), estimates, alpha=0.5)
plt.show()

#next, add analysis scaffold ( or warmup samp), use z score filter and stochastic confidence bands
