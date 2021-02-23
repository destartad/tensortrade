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

print(log_entry)