"""
s = "{0:." + str("5") + "f}" + " {1}"
s = s.format(0.2, "EURUSD")

print(s)"""

"""
from decimal import Decimal, ROUND_DOWN

print(Decimal(10)**-5)
"""

from itertools import product
from typing import Union, List, Any

a=[1,2]
b=2
c=3
x = product(a)
x = list(x)

