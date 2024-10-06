import numpy as np
import cvxpy as cp

def single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold):
    # Number of assets
    n = len(r_t)

    # Decision variable: trade vector z_t
    z_t = cp.Variable(n)

    # Transaction and holding cost
    trade_cost = phi_trade(z_t)
    hold_cost = phi_hold(w_t + z_t)

    # Define the risk measure: Assume a simple quadratic risk model
    sigma_t = np.eye(n)  # Placeholder for the covariance matrix (to be replaced with real data)
    risk = cp.quad_form(w_t + z_t, sigma_t)

    # Objective function: maximize return minus risk and costs
    objective = cp.Maximize(r_t.T @ z_t - gamma * risk - trade_cost - hold_cost)

    # Constraints: Self-financing without explicitly setting trade_cost and hold_cost to zero
    # Allow the portfolio weights to sum up to 1 after trades (budget constraint)
    constraints = [cp.sum(w_t + z_t) == 1] 

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem if the problem adheres to DCP rules
    if(problem.is_dcp()):
        problem.solve()
        return z_t.value
    else:
        print("Error: problem with inputted values was not DCP")
        return -1

# Test Case 1: High risk aversion
r_t = np.array([0.05, 0.07, 0.02])  # Expected returns
w_t = np.array([0.3, 0.4, 0.3])  # Current portfolio weights
gamma = 5.0  # High risk aversion parameter
phi_trade = lambda z: cp.norm(z, 1) * 0.01  # Trading cost
phi_hold = lambda w: cp.norm(w, 2) * 0.005  # Holding cost
optimal_trades = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
print("Test Case 1 (High risk aversion):", optimal_trades)

# Test Case 2: Low expected returns and low risk aversion
r_t = np.array([0.01, 0.03, 0.02])  # Lower expected returns
w_t = np.array([0.4, 0.4, 0.2])  # Current portfolio weights
gamma = 0.1  # Low risk aversion parameter
phi_trade = lambda z: cp.norm(z, 1) * 0.02  # Increased trading cost
phi_hold = lambda w: cp.norm(w, 2) * 0.002  # Lower holding cost
optimal_trades = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
print("Test Case 2 (Low returns, low risk aversion):", optimal_trades)

# Test Case 3: Different portfolio weights and moderate risk aversion
r_t = np.array([0.08, 0.05, 0.03])  # Higher expected return for asset 1
w_t = np.array([0.5, 0.3, 0.2])  # Different portfolio weights
gamma = 1.0  # Moderate risk aversion parameter
phi_trade = lambda z: cp.norm(z, 1) * 0.015  # Different trading cost
phi_hold = lambda w: cp.norm(w, 2) * 0.01  # Higher holding cost
optimal_trades = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
print("Test Case 3 (Different weights, moderate risk aversion):", optimal_trades)

# Test Case 4: Zero risk aversion (focus on maximizing return without considering risk)
r_t = np.array([0.04, 0.06, 0.03])  # Balanced expected returns
w_t = np.array([0.3, 0.3, 0.4])  # Different portfolio weights
gamma = 0.0  # Zero risk aversion parameter
phi_trade = lambda z: cp.norm(z, 1) * 0.005  # Lower trading cost
phi_hold = lambda w: cp.norm(w, 2) * 0.003  # Lower holding cost
optimal_trades = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
print("Test Case 4 (Zero risk aversion):", optimal_trades)

# Test Case 5: Very low trading and holding costs
r_t = np.array([0.03, 0.04, 0.06])  # Higher return for asset 3
w_t = np.array([0.3, 0.5, 0.2])  # Current portfolio weights
gamma = 0.7  # Medium risk aversion
phi_trade = lambda z: cp.norm(z, 1) * 0.001  # Very low trading cost
phi_hold = lambda w: cp.norm(w, 2) * 0.001  # Very low holding cost
optimal_trades = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
print("Test Case 5 (Low costs):", optimal_trades)

# Test Case 6: Increased transaction and holding costs
r_t = np.array([0.05, 0.04, 0.03])  # Moderate expected returns
w_t = np.array([0.25, 0.5, 0.25])  # Current portfolio weights
gamma = 0.8  # Risk aversion parameter
phi_trade = lambda z: cp.norm(z, 1) * 0.03  # Higher trading cost
phi_hold = lambda w: cp.norm(w, 2) * 0.02  # Higher holding cost
optimal_trades = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
print("Test Case 6 (High costs):", optimal_trades)