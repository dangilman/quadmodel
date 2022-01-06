import numpy as np
from quadmodel.inference.sample_prior import sample_from_prior

def setup_source(source_model, kwargs_source_priors):

    if source_model == 'GAUSSIAN':
        source_fwhm_pc = sample_from_prior(kwargs_source_priors['source_fwhm_pc'])
    elif source_model == 'DOUBLE_GAUSSIAN':
        raise Exception('not yet implemented')
    else:
        raise Exception('oops')

    return source_model, source_fwhm_pc
