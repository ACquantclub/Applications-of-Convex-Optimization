import sys
import os
import pytest
import numpy as np
from datetime import datetime

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from Portfolio import Portfolio
from single_period_optimization import single_period_optimization

# Fixture to create a sample portfolio with some trades
@pytest.fixture
def sample_portfolio1():
    #declare and initialize portfolio
    portfolio = Portfolio() #basic portfolio

    #trades vector to trade on
    trades = [
        Trade("AAPL", 100, 150.0, datetime(2023, 1, 1), TradeType.BUY),
        Trade("GOOGL", 50, 2000.0, datetime(2023, 1, 2), TradeType.BUY),
        Trade("MSFT", 75, 300.0, datetime(2023, 1, 3), TradeType.BUY)
    ]
    for trade in trades:
        portfolio.execute_trade(trade)
    return portfolio

@pytest.fixture
def sample_portfolio2():
    #declare and initialize portfolio
    portfolio = Portfolio() #basic portfolio

    #trades vector to trade on
    trades = [
        Trade("AAPL", 100, 150.0, datetime(2023, 1, 1), TradeType.BUY),
        Trade("GOOGL", 50, 2000.0, datetime(2023, 1, 2), TradeType.BUY),
        Trade("MSFT", 75, 300.0, datetime(2023, 1, 3), TradeType.BUY),
        Trade("AAPL", 100, 150.0, datetime(2023, 1, 1), TradeType.SELL),
        Trade("GOOGL", 50, 2000.0, datetime(2023, 1, 2), TradeType.SELL),
    ]
    for trade in trades:
        portfolio.execute_trade(trade)
    return portfolio

@pytest.fixture
def sample_portfolio3():
    #declare and initialize portfolio
    portfolio = Portfolio() #basic portfolio

    #trades vector to trade on
    trades = [
        Trade('AAPL', 10, 150.0, datetime.now(), TradeType.BUY),
        Trade('GOOGL', 5, 2500.0, datetime.now() - timedelta(days=1), TradeType.BUY),
        Trade('MSFT', 8, 300.0, datetime.now() - timedelta(days=2), TradeType.SELL),
        Trade('TSLA', 15, 800.0, datetime.now() - timedelta(days=3), TradeType.BUY),
        Trade('AMZN', 3, 3500.0, datetime.now() - timedelta(days=4), TradeType.SELL),
        Trade('NFLX', 20, 500.0, datetime.now() - timedelta(days=5), TradeType.BUY),
        Trade('AAPL', 5, 155.0, datetime.now() - timedelta(days=6), TradeType.SELL),
        Trade('FB', 7, 320.0, datetime.now() - timedelta(days=7), TradeType.BUY),
        Trade('GOOGL', 2, 2550.0, datetime.now() - timedelta(days=8), TradeType.SELL),
        Trade('TSLA', 10, 850.0, datetime.now() - timedelta(days=9), TradeType.BUY)
    ]
    for trade in trades:
        portfolio.execute_trade(trade)
    return portfolio

def high_gamma(self, portfolio):
    r_t = np.array([0.05, 0.07, 0.02])
    w_t = portfolio.get_weights()
    gamma = 4.0
    phi_hold = portfolio.phi_hold()
    phi_trade = portfolio.phi_trade()

    spo = single_period_optimization(r_t, w_t, gamme, phi_trade, phi_hold)
    self.assertEqual(Portfolio.single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold), spo)

def low_gamma(self):
    r_t = np.array([0.05, 0.07, 0.02])
    w_t = portfolio.get_weights()
    gamma = 0.1
    phi_hold = portfolio.phi_hold()
    phi_trade = portfolio.phi_trade()

    spo = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
    self.assertEqual(Portfolio.single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold), spo)
    
def med_gamma(self):
    r_t = np.array([0.05, 0.07, 0.02])
    w_t = portfolio.get_weights()
    gamma = 1.5
    phi_hold = portfolio.phi_hold()
    phi_trade = portfolio.phi_trade()

    spo = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
    self.assertEqual(Portfolio.single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold), spo)

# Run the tests
if __name__ == "__main__":
    pytest.main()