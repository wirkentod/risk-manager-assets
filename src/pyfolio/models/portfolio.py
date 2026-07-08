from typing import Dict
from .stock import Stock  # Import stock from models

class Portfolio:
    def __init__(self):
        self.components: Dict[Stock, float] = {}

    def add_stock(self, stock: Stock, weight: float):
        self.components[stock] = weight

    def calculate_performance(self) -> float:
        # Calculate the overall performance of the portfolio based on the performance of each stock and its weight
        # Do the calculation here, for example:
        return True
        #return sum(s.performance * p for s, p in self.components.items())