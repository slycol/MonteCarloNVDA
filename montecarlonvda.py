
#Monte Carlo Simulation of NVIDIA (NVDA) Stock Price

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("NVDA_history.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

prices = df["Adj Close"].values
dates = df["Date"].values

S0 = prices[-1]  # most recent available price as our starting point
last_date = pd.Timestamp(dates[-1])

print(f"Loaded {len(df):,} trading days of NVDA data")
print(f"Range: {pd.Timestamp(dates[0]).date()} to {last_date.date()}")
print(f"Most recent price in dataset: ${S0:,.2f} (as of {last_date.date()})")

log_returns = np.diff(np.log(prices))

mu_daily_full = np.mean(log_returns)
sigma_daily_full = np.std(log_returns)

window = 252 * 2
recent_returns = log_returns[-window:]
mu_daily_recent = np.mean(recent_returns)
sigma_daily_recent = np.std(recent_returns)

trading_days = 252

def annualize(mu_d, sigma_d):
    mu_annual = mu_d * trading_days
    sigma_annual = sigma_d * np.sqrt(trading_days)
    return mu_annual, sigma_annual

mu_full, sigma_full = annualize(mu_daily_full, sigma_daily_full)
mu_recent, sigma_recent = annualize(mu_daily_recent, sigma_daily_recent)

print("\n--- Parameters estimated from real data ---")
print(f"Full history (1999-2026):  annual return ~{mu_full*100:,.1f}%,  annual volatility ~{sigma_full*100:,.1f}%")
print(f"Last 2 years only:         annual return ~{mu_recent*100:,.1f}%,  annual volatility ~{sigma_recent*100:,.1f}%")
print("(NVDA's 27-year average is inflated by its early hyper-growth years,")
print(" so the recent window is arguably a more realistic forward-looking estimate.")
print(" Neither is a real forecast -- past returns don't guarantee future ones.)")

# We'll simulate using the RECENT window as our working assumption
mu = mu_recent
sigma = sigma_recent

T = 1.0                 # 1 year ahead
n_simulations = 10000
dt = T / trading_days

np.random.seed(42)
price_paths = np.zeros((n_simulations, trading_days + 1))
price_paths[:, 0] = S0

for t in range(1, trading_days + 1):
    Z = np.random.standard_normal(n_simulations)
    price_paths[:, t] = price_paths[:, t - 1] * np.exp(
        (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    )

final_prices = price_paths[:, -1]


mean_price = np.mean(final_prices)
median_price = np.median(final_prices)
pct_5, pct_25, pct_75, pct_95 = np.percentile(final_prices, [5, 25, 75, 95])
prob_above_current = np.mean(final_prices > S0) * 100
prob_double = np.mean(final_prices > 2 * S0) * 100
prob_halve = np.mean(final_prices < 0.5 * S0) * 100

print(f"\n--- Simulation Results ({n_simulations:,} runs, {T:.0f}-year horizon from {last_date.date()}) ---")
print(f"Starting price:            ${S0:,.2f}")
print(f"Mean projected price:      ${mean_price:,.2f}")
print(f"Median projected price:    ${median_price:,.2f}")
print(f"50% confidence range:      ${pct_25:,.2f}  to  ${pct_75:,.2f}")
print(f"90% confidence range:      ${pct_5:,.2f}  to  ${pct_95:,.2f}")
print(f"P(price > starting price):   {prob_above_current:.1f}%")
print(f"P(price doubles):            {prob_double:.1f}%")
print(f"P(price falls by half):      {prob_halve:.1f}%")


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sample_size = 200
for i in range(sample_size):
    axes[0].plot(price_paths[i], linewidth=0.6, alpha=0.4)
axes[0].axhline(S0, color="black", linestyle="--", linewidth=1, label=f"Start: ${S0:,.0f}")
axes[0].set_title(f"{sample_size} Simulated NVDA Paths (1 Yr, params from real data)")
axes[0].set_xlabel("Trading Day")
axes[0].set_ylabel("Price ($)")
axes[0].legend()

axes[1].hist(final_prices, bins=60, color="seagreen", edgecolor="white", alpha=0.85)
axes[1].axvline(mean_price, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_price:,.0f}")
axes[1].axvline(pct_5, color="gray", linestyle=":", linewidth=1.5, label=f"5th pct: ${pct_5:,.0f}")
axes[1].axvline(pct_95, color="gray", linestyle=":", linewidth=1.5, label=f"95th pct: ${pct_95:,.0f}")
axes[1].set_title("Distribution of Simulated Prices After 1 Year")
axes[1].set_xlabel("Price ($)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("nvda_monte_carlo_real_data.png", dpi=150)
print("\nChart saved to nvda_monte_carlo_real_data.png")


fig2, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, prices, linewidth=0.8, color="navy")
ax.set_title("NVDA Actual Historical Price (Adjusted Close, 1999-2026)")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("nvda_actual_history.png", dpi=150)
print("Chart saved to nvda_actual_history.png")