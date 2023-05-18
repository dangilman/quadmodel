import numpy as np
from scipy.interpolate import interp1d

def sample_from_prior(prior):

    prior_type = prior[0]
    if prior_type == 'FIXED':
        value = prior_type[1]
    elif prior_type == 'UNIFORM':
        value = np.random.uniform(prior[1], prior[2])
    elif prior_type == 'GAUSSIAN':
        value = np.random.normal(prior[1], prior[2])
    elif prior_type == 'CUSTOM_PDF':
        x, pdf = prior[1], prior[2]
        pdf *= np.max(pdf) ** -1
        pdf_interp = interp1d(x, pdf)
        while True:
            u = np.random.rand()
            value = np.random.uniform(np.min(x), np.max(x))
            p = pdf_interp(value)
            if p >= u:
                break
    return value
