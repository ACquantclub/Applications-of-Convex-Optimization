import numpy as np
import cvxpy as cp

def single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold):
    # Number of assets
    n = len(r_t)

    # Decision variable: trade vector z_t
    z_t = cp.Variable(n)

    # Transaction and holding cost (assume they are functions returning scalars)
    trade_cost = phi_trade(z_t)
    hold_cost = phi_hold(w_t + z_t)

    # Define the risk measure: Assume a simple quadratic risk model
    sigma_t = np.eye(n)  # Placeholder for the covariance matrix (to be replaced with real data)
    risk = cp.quad_form(w_t + z_t, sigma_t)

    # Objective function
    objective = cp.Maximize(r_t.T @ z_t - gamma * risk - trade_cost - hold_cost)

    # Constraints: Self-financing constraint
    constraints = [cp.sum(z_t) + trade_cost + hold_cost == 0]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    return z_t.value

# Example usage:
r_t = np.array([0.05, 0.07, 0.02])  # Expected returns
w_t = np.array([0.3, 0.4, 0.3])  # Current portfolio weights
gamma = 0.5  # Risk aversion parameter

# Placeholder cost functions
phi_trade = lambda z: cp.norm(z, 1) * 0.01  # Assume a simple linear cost for trading
phi_hold = lambda w: cp.norm(w, 2) * 0.005  # Assume a quadratic cost for holding

# Call the function
optimal_trades = single_period_optimization(r_t, w_t, gamma, phi_trade, phi_hold)
print("Optimal trade vector:", optimal_trades) 