#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:06:53 2024
using monte carlo simulation to estimate new data error
the law of large number ensures that as the number of samples increase,
the average of the result from the random samples will converge to the expected value

monte carlo simulation
1 random sampling: relies on generating random variables according to a specific probability distribution
2 repetiion: simulation is run multiple times to account for variability and randomness
is a powerful tool for estimating the probability distribution of outcomes
@author: chengu
"""

import numpy as np
#number of samples
num_sample = 10000
#generate samples for x_star, y_star
x_star_sample = np.random.normal(0,1,num_sample)
y_star_sample = 3*x_star_sample+np.random.normal(0,1,num_sample)
y_pred_sample = 2.5*x_star_sample+0.5
squared_error = (y_pred_sample-y_star_sample)**2
E_new = np.mean(squared_error)
print(squared_error.size)
print(E_new)