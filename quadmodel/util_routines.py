from multiprocessing.pool import Pool

def transform_single_likelihood(interpolated_likelihood, cross_section_samples, lambda_samples, sigma_sub_samples,
                                interp_collapse_68, interp_collapse_89, interp_collapse_910, nproc=8):

    args = []
    for i in range(0, cross_section_samples.shape[0]):
        point = tuple(cross_section_samples[i, :])
        args.append([interpolated_likelihood, point, interp_collapse_68, interp_collapse_89, interp_collapse_910,
                    lambda_samples[i], sigma_sub_samples[i]])

    pool = Pool(nproc)
    prob = pool.map(_like_from_point, args)
    pool.close()
    return prob

def _like_from_point(args):

    (interpolated_likelihood, point, interp_collapse_68, interp_collapse_89,
     interp_collapse_910, lambda_sample, sigma_sub_sample) = args
    f_68, f_89, f_910 = float(interp_collapse_68(point)), float(interp_collapse_89(point)), float(
        interp_collapse_910(point))
    new_point = (f_68, f_89, f_910, lambda_sample, sigma_sub_sample)
    p = interpolated_likelihood(new_point)
    return p
