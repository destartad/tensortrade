"""
s = "{0:." + str("5") + "f}" + " {1}"
s = s.format(0.2, "EURUSD")

print(s)"""

"""
from decimal import Decimal, ROUND_DOWN

print(Decimal(10)**-5)


from itertools import product
from typing import Union, List, Any

a=[1,2]
b=2
c=3
x = product(a)
x = list(x)
"""
'''
class Student(object):
    def __init__(self, score=0):
        self._score = score
    
    @property    
    def score(self):
        print("getting score")
        return self._score
    
    @score.setter
    def score(self, value):
        print("setting score")
        if not isinstance(value, int):
            raise ValueError("score must be an integer!")           
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
        
s = Student(60)
s.score
print("=====================")
s.score = 88
s.score

trade_sizes = 10
trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]
print(trade_sizes)'''
"""
from decimal import *


x=Decimal('4.2123412').quantize(Decimal('0.00'), rounding=ROUND_DOWN)
print(x)

from decimal import *

y:Decimal = 0.00
print(y)
"""
"""
import os
import sys
import logging
import importlib

from abc import abstractmethod
from datetime import datetime
from typing import Union, Tuple
from collections import OrderedDict


date_format = "%Y-%m-%d %H:%M:%S %p"

log_entry = f"[{datetime.now().strftime(date_format)}]"


filled_time=datetime(2021,2,16,5,10,00)
current_time=datetime(2021,3,2,4,10,00)
z="2019-11-08 06:36:00 AM"
deal = datetime.strptime(z,"%Y-%m-%d %H:%M:%S %p" )

def calc_swap_weekday(filled_time,current_time):
    
        date_diff = (current_time.date() - filled_time.date()).days
        z = date_diff//7
        y = current_time.weekday()
        if date_diff <= 7:
            return date_diff
        elif y == 0 or y == 1:
            return date_diff
        elif y == 2 or y == 3 or y == 4:
            return date_diff + 2
        elif y == 5 or y == 6:
            return date_diff

x = calc_swap_weekday(filled_time, current_time)
print(deal)
"""

from ray import tune


def objective(step, alpha, beta):
    return (0.1 + alpha * step / 100)**(-1) + beta * 0.1


def training_function(config):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = objective(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)


analysis = tune.run(
    training_function,
    config={
        "alpha": tune.grid_search([0.001, 0.01, 0.1]),
        "beta": tune.choice([1, 2, 3])
    })

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
