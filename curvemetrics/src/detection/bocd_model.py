from functools import partial

from .bocd.bayesian_changepoint_detection.online_likelihoods import StudentT
from .bocd.bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
from .bocd.bayesian_changepoint_detection.hazard_functions import constant_hazard

def _constant_hazard(lambda_):
    return partial(constant_hazard, lambda_)

def _log_likelihood_class(alpha, beta, kappa, mu):
    return StudentT(alpha=alpha, beta=beta, kappa=kappa, mu=mu)

def parse_args(**kwargs):
    lambda_ = kwargs.get("lambda_", 100)
    alpha = kwargs.get("alpha", 0.1)
    beta = kwargs.get("beta", 0.01)
    kappa = kwargs.get("kappa", 1)
    mu = kwargs.get("mu", 0)
    return lambda_, alpha, beta, kappa, mu

def bocd(data, hazard_function=None, log_likelihood_class=None, **kwargs):
    """
    Wrapper for calling `online_changepoint_detection` from the `bayescd` package.

    Credits: https://github.com/hildensia/bayesian_changepoint_detection/tree/master

    :param data: data to be analyzed
    :param hazard_function: hazard function to be used
    :param ll: likelihood function to be used

    :return R: the probability at time step t that the last sequence is already s time steps long
    :return maxes: the argmax on column axis of matrix R (growth probability value) for each time step
    """

    lambda_, alpha, beta, kappa, mu = parse_args(**kwargs)

    hazard_function = hazard_function or _constant_hazard(lambda_)
    log_likelihood_class = log_likelihood_class or _log_likelihood_class(alpha, beta, kappa, mu)

    R, maxes = online_changepoint_detection(
        data, hazard_function, log_likelihood_class
    )

    return R, maxes
