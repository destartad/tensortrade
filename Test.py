"""
s = "{0:." + str("5") + "f}" + " {1}"
s = s.format(0.2, "EURUSD")

print(s)"""


from decimal import Decimal, ROUND_DOWN

print(Decimal(10)**-5)