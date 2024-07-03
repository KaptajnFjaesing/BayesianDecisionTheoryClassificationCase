import numpy as np
import pymc as pm

# Generate dummy data
np.random.seed(42)
N = 1000
X = np.random.randn(N, 2)
true_coeffs = [2.5, -1.0]
true_intercept = 1.0
true_sigma = 0.5
y = np.dot(X, true_coeffs) + true_intercept + np.random.randn(N) * true_sigma

# Create PyMC model
with pm.Model() as model:
    # Priors
    coeffs = pm.Normal('coeffs', mu=0, sigma=10, shape=2)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Likelihood
    likelihood = pm.Normal('y', mu=intercept + pm.math.dot(X, coeffs), sigma=sigma, observed=y)

    # Sample from the posterior
    trace = pm.sample(tune = 500, draws = 5000, chains = 4, cores = 1)

# Print summary statistics
print(pm.summary(trace))
