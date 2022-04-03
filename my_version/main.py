# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:48:44 2022

@author: ZongSing_NB2
"""

from NSGAII import NSGAII
import benchmark

import time

optimizer = NSGAII(benchmark=benchmark.sch(),
                   P=100,
                   G=100,
                   tour_k=2)
optimizer.opt()
optimizer.plot(title='sch',
               xlabel='objective 1',
               ylabel='objective 2')
