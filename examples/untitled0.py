# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:23:15 2019

@author: evrardgarcelon
"""

import numpy as np
import pylab as plt

x = np.linspace(0,100)
y = np.linspace(0,50)

c_1 = 1
a = np.array([1,-1])
f = lambda c,d : c_1*np.sqrt(c+d) + a[0]*c + a[1]*d

X,Y = np.meshgrid(x,y)

res = f(X,Y)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x,y,res)

plt.show()