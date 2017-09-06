#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from ipdb import set_trace as b
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

np.random.seed(1)


# --- Define your problem
def f(x):
    return 0.1 * x + (np.sin(x)+1)


def f_(x,sig=0.2):
    """
    f() with noise
    """
    return np.random.normal(f(x), sig)


def acq(mean, sig, beta=3):
    """
    Acquisition function
    (Upper Confidence Bound)
    $ x_t = argmax\ \mu_{t-1} + \sqrt{\beta_t} \sigma_{t-1}(x) $
    """
    return np.argmax(mean + sig * np.sqrt(beta))


def plot(x_grid, X, y, y_pred, sigma):

    plt.close()
    fig = plt.figure()
    plt.plot(x_grid, f(x_grid), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x_grid, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x_grid, x_grid[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma,(y_pred + 1.96 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(0, 4)
    plt.legend(loc='upper left')
    plt.savefig('fig/fig%02d.png' % (i))


#
# Application entry point.
#
if __name__ == '__main__':

    # Parameter grid (parameter space).
    x = np.atleast_2d(np.linspace(0, 10, 100)).T
    
    # Initial value
    X = np.atleast_2d([0., 10.]).T
    y = f(X).ravel()
    
    prev_idx = None
    nitr     = 20
    decay    = 0.9
    beta     = 200.
    
    for i in range(nitr):
    
        gp = GaussianProcessRegressor()
        gp.fit(X, y)
    
        posterior_mean, posterior_sig = gp.predict(x, return_std=True)
        plot(x, X, y, posterior_mean, posterior_sig)
    
        idx = acq(posterior_mean, posterior_sig, beta=beta)
        
        prev_idx = idx
    
        # update
        X = np.atleast_2d([np.r_[X[:, 0], x[idx]]]).T
        y = np.r_[y, f(x[idx])]
        
        beta *= decay    
    
    
    os.system('convert -delay 100 -loop 0 fig/*.png bo.gif')
    
