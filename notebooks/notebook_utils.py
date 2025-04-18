import numpy as np

def bootstrap_mean_and_ci(data, n_bootstrap=1000, ci=95):
    """Bootstraps mean and confidence interval across axis 0, handling nan values"""
    n_samples = data.shape[0]
    boot_means = np.empty((n_bootstrap, data.shape[1]))

    for i in range(n_bootstrap):
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_sample = data[sample_indices]
        boot_means[i] = np.nanmean(boot_sample, axis=0)

    lower_bound = np.nanpercentile(boot_means, (100 - ci) / 2, axis=0)
    upper_bound = np.nanpercentile(boot_means, 100 - (100 - ci) / 2, axis=0)
    mean_curve = np.nanmean(boot_means, axis=0)
    return mean_curve, lower_bound, upper_bound