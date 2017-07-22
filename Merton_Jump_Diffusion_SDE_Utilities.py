import numpy as np
from numpy import random as nrand
import math
import random

class ModelParameters:
    """
    Encapsulates model parameters
    """
    
    def __init__(self,
                 all_time, all_delta, all_sigma, gbm_mu,
                 jumps_lamda=0.0, jumps_sigma=0.0, jumps_mu=0.0):
        
        # This is the amount of time to simulate for
        self.all_time = all_time
        
        # This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
        self.all_delta = all_delta
        
        # This is the volatility of the stochastic processes
        self.all_sigma = all_sigma
        
        # This is the annual drift factor for geometric brownian motion
        self.gbm_mu = gbm_mu
        
        # This is the probability of a jump happening at each point in time
        self.lamda = jumps_lamda
        
        # This is the volatility of the jump size
        self.jumps_sigma = jumps_sigma
        
        # This is the average jump size
        self.jumps_mu = jumps_mu
        

def random_model_params():
    return ModelParameters(
        # Fixed Parameters
        all_time=2000,
        all_delta=0.00396825396,
        
        # Random Parameters
        all_sigma = nrand.uniform(0.001,0.2),        
        gbm_mu = nrand.uniform(-1,1),
        jumps_lamda=nrand.uniform(0.0001,0.025),
        jumps_sigma=nrand.uniform(0.001, 0.2),
        jumps_mu=nrand.uniform(-0.5,0.5),
    )

def brownian_motion_log_returns(param):
    """
    This method returns a Wiener process. The Wiener process is also called Brownian motion. For more information
    about the Wiener process check out the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process
    :param param: the model parameters object
    :return: brownian motion log returns
    """
    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=param.all_time)

def geometric_brownian_motion_log_returns(param):
    """
    This method constructs a sequence of log returns which, when exponentiated, produce a random Geometric Brownian
    Motion (GBM). GBM is the stochastic process underlying the Black Scholes options pricing formula.
    :param param: model parameters object
    :return: returns the log returns of a geometric brownian motion process
    """
    assert isinstance(param, ModelParameters)
    wiener_process = np.array(brownian_motion_log_returns(param))
    sigma_pow_mu_delta = (param.gbm_mu - 0.5 * math.pow(param.all_sigma, 2.0)) * param.all_delta
    return wiener_process + sigma_pow_mu_delta

def jump_diffusion_process(param):
    """
    This method produces a sequence of Jump Sizes which represent a jump diffusion process. These jumps are combined
    with a geometric brownian motion (log returns) to produce the Merton model.
    :param param: the model parameters object
    :return: jump sizes for each point in time (mostly zeroes if jumps are infrequent)
    """
    assert isinstance(param, ModelParameters)
    s_n = time = 0
    small_lamda = -(1.0 / param.lamda)
    jump_sizes = np.zeros(param.all_time)
    while s_n < param.all_time:
        s_n += small_lamda * math.log(random.uniform(0, 1))
        for j in range(0, param.all_time):
            if time * param.all_delta <= s_n * param.all_delta <= (j + 1) * param.all_delta:
                jump_sizes[j] += random.normalvariate(param.jumps_mu, param.jumps_sigma)
                break
        time += 1
    return jump_sizes

def geometric_brownian_motion_jump_diffusion_log_returns(param):
    """
    This method constructs combines a geometric brownian motion process (log returns) with a jump diffusion process
    (log returns) to produce a sequence of gbm jump returns.
    :param param: model parameters object
    :return: returns a GBM process with jumps in it
    """
    assert isinstance(param, ModelParameters)
    jump_diffusion = jump_diffusion_process(param)
    geometric_brownian_motion = geometric_brownian_motion_log_returns(param)
    return np.add(jump_diffusion, geometric_brownian_motion)