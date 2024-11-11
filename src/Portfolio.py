import numpy as np
from typing import List, Dict
from Trade import Trade, TradeType
from single_period_optimization import single_period_optimization

class Portfolio:
    def __init__(self):
        """
        Initialize a portfolio with empty holdings, trades, symbols, and vectors for holdings and weights.
        - holdings: Dictionary to store the quantity of each stock symbol.
        - trades: List to store all trade transactions.
        - symbols: List to keep track of all unique stock symbols in the portfolio.
        - holdings_vector: Numpy array representing quantities of each stock in the same order as symbols.
        - weights_vector: Numpy array representing the proportion of each stock in the portfolio.
        """
        self.holdings: Dict[str, int] = {}
        self.trades: List[Trade] = []
        self.symbols: List[str] = []
        self.holdings_vector: np.ndarray = np.array([])
        self.weights_vector: np.ndarray = np.array([])

    def execute_trade(self, trade: Trade):
        """
        Executes a trade (buy or sell) by updating holdings and adjusting portfolio weights accordingly.
        - trade: A Trade object containing trade details (symbol, quantity, trade_type).
        """
        # Add the trade to the list of executed trades
        self.trades.append(trade)
        
        # If the stock symbol is new, add it to the symbols list and extend the vectors for holdings and weights
        if trade.symbol not in self.symbols:
            self.symbols.append(trade.symbol)
            self.holdings_vector = np.append(self.holdings_vector, 0)  # Append zero to holdings vector for new symbol
            self.weights_vector = np.append(self.weights_vector, 0)  # Append zero to weights vector for new symbol
        
        # Find the index of the stock symbol in the symbols list
        symbol_index = self.symbols.index(trade.symbol)
        
        # Handle a BUY trade: increase the holdings of the stock
        if trade.trade_type == TradeType.BUY:
            # Update holdings dictionary by adding the quantity bought
            self.holdings[trade.symbol] = self.holdings.get(trade.symbol, 0) + trade.quantity
            self.holdings_vector[symbol_index] += trade.quantity  # Update the holdings vector for the symbol
        
        # Handle a SELL trade: decrease the holdings of the stock
        elif trade.trade_type == TradeType.SELL:
            # Check if there are enough shares to sell, otherwise raise an error
            if trade.symbol not in self.holdings or self.holdings[trade.symbol] < trade.quantity:
                raise ValueError(f"Not enough {trade.symbol} shares to sell")
            
            # Reduce the holdings by the quantity sold
            self.holdings[trade.symbol] -= trade.quantity
            self.holdings_vector[symbol_index] -= trade.quantity  # Update the holdings vector for the symbol
            
            # If all shares of the stock are sold, remove it from the holdings dictionary
            if self.holdings[trade.symbol] == 0:
                del self.holdings[trade.symbol]
        
        # After the trade, update the portfolio weights
        self.update_weights()

    def update_weights(self):
        """
        Updates the portfolio weights based on the current holdings.
        """
        total_holdings = np.sum(self.holdings_vector)  # Calculate total number of shares across all holdings
        
        if total_holdings > 0:
            # If total holdings are greater than zero, calculate the weight of each stock
            self.weights_vector = self.holdings_vector / total_holdings
        else:
            # If there are no holdings, set all weights to zero
            self.weights_vector = np.zeros(len(self.symbols))

    def get_weights(self) -> Dict[str, float]:
        """
        Returns a dictionary of stock symbols and their corresponding weights in the portfolio.
        """
        return {symbol: weight for symbol, weight in zip(self.symbols, self.weights_vector)}

    def total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculates the total market value of the portfolio based on current stock prices.
        """
        # Create a vector of current prices in the same order as the symbols list
        prices_vector = np.array([current_prices[symbol] for symbol in self.symbols])
        # Return the dot product of the holdings vector and prices vector to get the portfolio's total value
        return np.dot(self.holdings_vector, prices_vector)
    
    # Define the cost functions
    def phi_trade(x):
        return cp.sum(cp.abs(z))  # Example transaction cost function

    def phi_hold(x):
        return cp.sum(w ** 2)  # Example holding cost function


    def calculate_spo(Self):

        r_t = np.array([0.05, 0.07, 0.02]) #replace with actual data, temp data to test function
        w_t = self.holdings #holdings array of portfolio
        gamma = 0.5 #replace with actual data 
        return single_period_optimization(r_t, w_t, gamme, phi_trade, phi_hold)
