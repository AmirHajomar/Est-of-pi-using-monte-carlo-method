import monte_carlo_pi
import matplotlib
from matplotlib import pyplot as plt
import math
print("Enter the desired number of trials for π estimation:")
num_trials = input()
num_trials = int(num_trials)
# here, we'll run the monte carlo sim as many times as specified by the user, 
# and store the results in the dictionary "estimates"
estimates = []
for n in range(num_trials):
    pi_estimate, x, y, inside = monte_carlo_pi.estimate_pi(10000, seed=None)
    estimates.append(pi_estimate)
# Now we can analyze through quite a few statistical methods, we'll intuitively start first with mean
# then compare the mean of these raw samples to that of the filtered samples after the filtered samples 'stabilize' 

mean_estimate = sum(estimates) / len(estimates)
print(f"Mean of π estimates after {num_trials} trials: {mean_estimate}")


#next, add analysis scaffold ( or warmup samp), use z score filter and stochastic confidence bands
start = 0.1 * num_trials #The 10% Rule: A sample should generally not exceed 10% of the total population to ensure independence of trials.
start = int(start)
estimates_run = []
for n in range(0, start):
    estimates_run.append(estimates[n])
for n in range(0, num_trials):
    current_mean = sum(estimates_run) / len(estimates_run)
    current_std = (sum([(est - current_mean) ** 2 for est in estimates_run]) / len(estimates_run)) ** 0.5
    z_score = (estimates[n] - current_mean) / current_std if current_std != 0 else 0 
    if abs(z_score) < 3:  # Using a z-score threshold x to filter. (rmbr 68 95 99.7 rule)
        estimates_run.append(estimates[n])
#clc stcstc bands
means = []
upper1 = []
lower1 = []
upper2 = []
lower2 = []
s = 0.0
s2 = 0.0
for i, e in enumerate(estimates_run, start=1):
    s += e
    s2 += e * e

    mean = s / i
    var = (s2 / i) - mean**2
    std = math.sqrt(var) if var > 0 else 0.0
    se = std / math.sqrt(i)

    means.append(mean)
    upper1.append(mean + 1.96 * se)
    lower1.append(mean - 1.96 * se)
    upper2.append(mean + 1.04 * se)
    lower2.append(mean - 1.04 * se)
# stochastic confidence bands computed 95% and 70%, now we can plot the running mean of the filtered estimates 

# Finally, we can plot the running mean of the π estimates to visualize convergence
plt.plot(means, label="Running mean (filtered)")
plt.fill_between(
    range(len(means)),
    lower1,
    upper1,
    alpha=0.25,
    label="95% confidence band"
)
plt.fill_between(
    range(len(means)),
    lower2,
    upper2,
    alpha=0.25,
    label="70% confidence band"
)
#cut = 0.1*len(estimates_run)
#cut = int(cut)
pi = 3.141592653589793
pi_hat = sum(estimates_run[start:]) / len(estimates_run[start:])
raw_per = (abs((mean_estimate) - pi)/pi) * 1
fil_per = (abs((pi_hat) - pi)/pi) * 1
print(f"Percentage error of unfiltered mean estimate: {raw_per}%")
print(f"Percentage error of filtered mean estimate: {fil_per}%")
if (fil_per) < (raw_per):
    print("Filtering improved the estimate.")
else:
    print("Filtering did not improve the estimate.")
print(f"Analysis complete.")

print(f"Final π estimate after filtering and convergence : {pi_hat}")
plt.axhline(3.141592653589793, linestyle="--", label="True π")
plt.axhline(pi_hat, linestyle="--", label="filtered pi mean", color="green")
plt.axhline(mean_estimate, linestyle="--", label="mean of unfiltered estimates",color = "orange")
plt.xlabel("Accepted samples (after filter)")
plt.ylabel("π estimate")
plt.title(f"Filtered Monte Carlo π with Confidence Envelope\nFinal π estimate after filtering and convergence): {pi_hat}")
plt.legend()
plt.show()
