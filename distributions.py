import tensorflow as tf
import edward as ed
from edward.models import Normal
from edward.models.random_variables import TransformedDistribution
def logNormal(loc,scale):
    return TransformedDistribution(
        distribution=Normal(loc=loc, scale=scale),
        bijector= tf.contrib.distributions.bijectors.Exp(),
        name = "logNormal")

def covariance_2dim(params):
    sigma11 = logNormal(loc=params[0][0], scale=params[0][1])
    sigma22 = logNormal(loc=params[1][0], scale=params[1][1])
    sigma21 =  Normal(loc=params[2][0],scale=params[2][1])
    elements = [sigma11, sigma22, sigma21]
    tril = sigma11*[[1.0,0.0],[0.0,0.0]]+sigma22*[[0.0,0.0],[0.0,1.0]]+sigma21*[[0.0,0.0],[1.0,0.0]]
    return elements, tril

def covariance_2dim2(params):
    elements = Normal(loc=params[0], scale=params[1])

    tril = tf.exp(elements[0])*[[1.0,0.0],[0.0,0.0]]+tf.exp(elements[1])*[[0.0,0.0],[0.0,1.0]]+elements[2]*[[0.0,0.0],[1.0,0.0]]
    return elements, tril

def covariance_2dim3(diag_loc, diag_scale, nondiag_loc, nondiag_scale):
    sigma11 = logNormal(loc=diag_loc[0], scale=diag_scale)
    sigma22 = logNormal(loc=diag_loc[1], scale=diag_scale)
    sigma21 =  Normal(loc=nondiag_loc, scale=nondiag_scale)
    elements = [sigma11, sigma22, sigma21]
    tril = sigma11*[[1.0,0.0],[0.0,0.0]]+sigma22*[[0.0,0.0],[0.0,1.0]]+sigma21*[[0.0,0.0],[1.0,0.0]]
    return elements, tril