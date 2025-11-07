import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

np.random.seed(42)

# Parameters for grid
maturities = np.concatenate((np.linspace(0.03, 0.25, 6), np.linspace(0.5, 2.0, 6)))  # in years
moneyness = np.linspace(0.6, 1.4, 81)  # K / F (spot normalized to 1)

# convert to log-moneyness k = ln(K/F)
k_grid = np.log(moneyness)

# SVI-like param functions (vary with T)
def a_of_T(T):
    # Base total variance offset (decreases slowly with T)
    return 0.02 + 0.01 * np.exp(-3 * T)

def b_of_T(T):
    # amplitude of smile (decreases with T)
    return 0.3 / (1.0 + 2.0 * T)

def rho_of_T(T):
    # skew parameter (slightly time-dependent)
    return -0.5 + 0.2 * np.tanh(2 * (0.5 - T))

def m_of_T(T):
    # shift of the smile (small)
    return -0.02 * np.sin(2 * np.pi * T / 1.5)

def sigma_of_T(T):
    # smoothness parameter
    return 0.10 + 0.06 * (1 - np.exp(-T*1.5))

def total_variance_svi(k, T):
    # SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    a = a_of_T(T)
    b = b_of_T(T)
    rho = rho_of_T(T)
    m = m_of_T(T)
    sigma = sigma_of_T(T)
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    # add a tiny positive floor and some maturity-dependent noise
    noise = 0.0002 * np.random.randn(*np.shape(k)) * (1 + 0.5 * np.exp(-T))
    w = np.maximum(w + noise, 1e-6)
    return w

# Build surface: implied vol = sqrt(total variance / T)
IV_surface = np.zeros((len(maturities), len(k_grid)))
for i, T in enumerate(maturities):
    w = total_variance_svi(k_grid, T)
    iv = np.sqrt(w / T)
    IV_surface[i, :] = iv

# Pack into a DataFrame for convenience (rows: maturity, cols: moneyness)
moneyness_labels = [f"{m:.2f}" for m in moneyness]
df_iv = pd.DataFrame(IV_surface, index=np.round(maturities,3), columns=moneyness_labels)

# Plot 1: 3D surface
fig1 = plt.figure(figsize=(12, 7))
ax = fig1.add_subplot(111, projection='3d')
K_mesh, T_mesh = np.meshgrid(moneyness, maturities)
surf = ax.plot_surface(K_mesh, T_mesh, IV_surface, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('Strike / Forward (K/F)')
ax.set_ylabel('Maturity (years)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Simulated Implied Volatility Surface (SVI-like)')
fig1.colorbar(surf, shrink=0.6, aspect=12, label='IV')


# Plot 3: Slices for selected maturities
selected_idx = [0, 3, 7, len(maturities)-1]  # early short, short, mid, long
fig3, ax3 = plt.subplots(figsize=(10, 5))
for idx in selected_idx:
    T = maturities[idx]
    ax3.plot(moneyness, IV_surface[idx, :], label=f"T={T:.2f}y", linewidth=2)
ax3.set_xlabel('Strike / Forward (K/F)')
ax3.set_ylabel('Implied Volatility')
ax3.set_title('IV Smile Slices for Selected Maturities')
ax3.legend()
ax3.grid(alpha=0.3)

# Show some summary
print("Simulated IV surface dimensions:", IV_surface.shape)
print("Maturities (years):", maturities)
print("Example IV at ATM (K/F=1.00):")
atm_col = np.argmin(np.abs(moneyness - 1.0))
for i, T in enumerate(maturities):
    if i % 3 == 0:
        print(f" T={T:.2f}y -> IV={IV_surface[i, atm_col]:.4f}")

plt.show()
