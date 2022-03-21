# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:48:44 2022

@author: ZongSing_NB2
"""

from NSGAII import NSGAII
import benchmark
import dimension
import bound

optimizer = NSGAII(fitness=benchmark.sch,
                   D=dimension.sch(),
                   P=100,
                   G=1000,
                   XMAX=bound.sch()[1],
                   XMIN=bound.sch()[0],
                   k=2)
optimizer.opt()
