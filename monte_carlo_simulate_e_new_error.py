#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:06:53 2024
using monte carlo simulation to estimate new data error

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