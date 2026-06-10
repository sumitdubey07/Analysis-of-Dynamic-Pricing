"""
Statistical Analysis of Dynamic Pricing – Ride-Hailing Platforms
analysis.py: Data generation, statistical hypothesis testing, regression analysis,
             and pricing elasticity visualisation reports.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Style ─────────────────────────────────────────────────────────────────────
BG       = "#0D1117"
PANEL    = "#161B22"
BORDER   = "#30363D"
TEXT     = "#E6EDF3"
MUTED    = "#8B949E"
TEAL     = "#2EC4B6"
CORAL    = "#E63946"
AMBER    = "#F4A261"
BLUE     = "#457B9D"
GREEN    = "#2D6A4F"
PURPLE   = "#7B2D8B"
PALETTE  = [TEAL, CORAL, AMBER, BLUE, GREEN, PURPLE]

def set_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.titlepad":     10,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT,
        "font.family":       "DejaVu Sans",
    })

def save_fig(name):
    path = f"{name}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

CITIES     = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]
TIME_SLOTS = ["Early Morning (5–8)", "Morning Rush (8–11)", "Midday (11–14)",
              "Afternoon (14–17)", "Evening Rush (17–20)", "Night (20–23)", "Late Night (23–5)"]
WEATHER    = ["Clear", "Cloudy", "Rain", "Heavy Rain"]
DAYS       = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def generate_dataset(n=5000):
    city        = np.random.choice(CITIES, n)
    day         = np.random.choice(DAYS, n)
    time_slot   = np.random.choice(TIME_SLOTS, n)
    weather     = np.random.choice(WEATHER, n, p=[0.50, 0.25, 0.18, 0.07])
    base_fare   = np.random.uniform(60, 180, n)
    is_weekend  = (pd.Series(day).isin(["Saturday","Sunday"])).values.astype(int)

    # surge multiplier driven by time + weather + weekend
    surge_base  = np.ones(n)
    surge_base += np.where(np.isin(time_slot, ["Morning Rush (8–11)", "Evening Rush (17–20)"]), 0.8, 0)
    surge_base += np.where(np.isin(time_slot, ["Late Night (23–5)"]), 0.4, 0)
    surge_base += np.where(weather == "Rain", 0.3, 0)
    surge_base += np.where(weather == "Heavy Rain", 0.6, 0)
    surge_base += np.where(is_weekend == 1, 0.2, 0)
    surge_mult  = (surge_base + np.random.normal(0, 0.15, n)).clip(1.0, 3.5).round(2)

    final_fare  = (base_fare * surge_mult + np.random.normal(0, 8, n)).clip(50, 900).round(2)

    # demand: elastic — higher surge → lower demand, modulated by time of day
    demand_base = np.where(np.isin(time_slot, ["Morning Rush (8–11)", "Evening Rush (17–20)"]), 420, 220)
    demand_base += np.where(weather == "Rain", 80, 0)
    demand_base += np.where(weather == "Heavy Rain", 120, 0)
    demand_base += np.where(is_weekend == 1, -60, 0)
    demand      = (demand_base - 95 * surge_mult + np.random.normal(0, 30, n)).clip(20, 600).astype(int)

    # cancellation rate rises with surge
    cancel_rate = (0.05 + 0.12 * (surge_mult - 1) + np.random.normal(0, 0.02, n)).clip(0, 0.55).round(3)

    # driver supply rises with surge (incentive)
    driver_supply = (80 + 55 * (surge_mult - 1) + np.random.normal(0, 10, n)).clip(20, 300).astype(int)

    wait_time   = (4 + 6 / driver_supply * 100 + np.random.normal(0, 1, n)).clip(1, 25).round(1)
    rating      = (4.5 - 0.15 * surge_mult + np.random.normal(0, 0.2, n)).clip(1, 5).round(1)

    df = pd.DataFrame({
        "City": city, "Day": day, "TimeSlot": time_slot,
        "Weather": weather, "IsWeekend": is_weekend,
        "BaseFare": base_fare.round(2), "SurgeMultiplier": surge_mult,
        "FinalFare": final_fare, "Demand": demand,
        "CancellationRate": cancel_rate, "DriverSupply": driver_supply,
        "WaitTime": wait_time, "UserRating": rating,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_stats(df):
    print("\n── Descriptive Statistics ──────────────────────────────────")
    cols = ["SurgeMultiplier", "FinalFare", "Demand", "CancellationRate", "WaitTime", "UserRating"]
    desc = df[cols].describe().round(3)
    print(desc.to_string())

    print(f"\n  Surge > 1.5x : {(df['SurgeMultiplier'] > 1.5).sum()} rides "
          f"({(df['SurgeMultiplier'] > 1.5).mean()*100:.1f}%)")
    print(f"  Surge > 2.0x : {(df['SurgeMultiplier'] > 2.0).sum()} rides "
          f"({(df['SurgeMultiplier'] > 2.0).mean()*100:.1f}%)")
    print(f"  Avg fare during rush hours : ₹"
          f"{df[df['TimeSlot'].isin(['Morning Rush (8–11)','Evening Rush (17–20)'])]['FinalFare'].mean():.2f}")
    print(f"  Avg fare during off-peak   : ₹"
          f"{df[~df['TimeSlot'].isin(['Morning Rush (8–11)','Evening Rush (17–20)'])]['FinalFare'].mean():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HYPOTHESIS TESTS
# ─────────────────────────────────────────────────────────────────────────────

def hypothesis_tests(df):
    results = {}
    print("\n── Hypothesis Testing ──────────────────────────────────────")

    # H1: Rush-hour demand vs off-peak demand
    rush   = df[df["TimeSlot"].isin(["Morning Rush (8–11)","Evening Rush (17–20)"])]["Demand"]
    offpk  = df[~df["TimeSlot"].isin(["Morning Rush (8–11)","Evening Rush (17–20)"])]["Demand"]
    t1, p1 = ttest_ind(rush, offpk, equal_var=False)
    results["H1_rush_vs_offpeak"] = {"t": round(t1,3), "p": round(p1,6),
                                      "significant": p1 < 0.05,
                                      "rush_mean": rush.mean(), "offpk_mean": offpk.mean()}
    print(f"\n  H1 – Rush vs Off-Peak Demand")
    print(f"     t={t1:.3f}, p={p1:.6f}  {'✅ Significant' if p1<0.05 else '❌ Not significant'}")

    # H2: Rainy vs clear weather surge multiplier
    rain   = df[df["Weather"].isin(["Rain","Heavy Rain"])]["SurgeMultiplier"]
    clear  = df[df["Weather"] == "Clear"]["SurgeMultiplier"]
    u2, p2 = mannwhitneyu(rain, clear, alternative="greater")
    results["H2_rain_vs_clear_surge"] = {"U": round(u2,3), "p": round(p2,6),
                                          "significant": p2 < 0.05,
                                          "rain_mean": rain.mean(), "clear_mean": clear.mean()}
    print(f"\n  H2 – Rain vs Clear Weather Surge (Mann-Whitney U)")
    print(f"     U={u2:.0f}, p={p2:.6f}  {'✅ Significant' if p2<0.05 else '❌ Not significant'}")

    # H3: Weekend vs weekday cancellation rate
    wknd  = df[df["IsWeekend"]==1]["CancellationRate"]
    wkdy  = df[df["IsWeekend"]==0]["CancellationRate"]
    t3, p3 = ttest_ind(wknd, wkdy, equal_var=False)
    results["H3_weekend_cancellation"] = {"t": round(t3,3), "p": round(p3,6),
                                           "significant": p3 < 0.05,
                                           "wknd_mean": wknd.mean(), "wkdy_mean": wkdy.mean()}
    print(f"\n  H3 – Weekend vs Weekday Cancellation Rate")
    print(f"     t={t3:.3f}, p={p3:.6f}  {'✅ Significant' if p3<0.05 else '❌ Not significant'}")

    # H4: High surge → lower user rating (Pearson)
    r4, p4 = pearsonr(df["SurgeMultiplier"], df["UserRating"])
    results["H4_surge_rating_corr"] = {"r": round(r4,4), "p": round(p4,6), "significant": p4 < 0.05}
    print(f"\n  H4 – Surge vs User Rating Correlation (Pearson)")
    print(f"     r={r4:.4f}, p={p4:.6f}  {'✅ Significant' if p4<0.05 else '❌ Not significant'}")

    # H5: Surge vs Demand elasticity (Spearman)
    rs5, ps5 = spearmanr(df["SurgeMultiplier"], df["Demand"])
    results["H5_surge_demand_spearman"] = {"rho": round(rs5,4), "p": round(ps5,6), "significant": ps5 < 0.05}
    print(f"\n  H5 – Surge vs Demand (Spearman ρ)")
    print(f"     ρ={rs5:.4f}, p={ps5:.6f}  {'✅ Significant' if ps5<0.05 else '❌ Not significant'}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. REGRESSION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def regression_analysis(df):
    print("\n── Regression Analysis ─────────────────────────────────────")

    # Simple linear: Surge → Demand
    X_lin = df[["SurgeMultiplier"]].values
    y_dem = df["Demand"].values
    lr = LinearRegression().fit(X_lin, y_dem)
    y_pred_lin = lr.predict(X_lin)
    r2_lin = r2_score(y_dem, y_pred_lin)
    print(f"\n  Linear Regression: Surge → Demand")
    print(f"     Coeff: {lr.coef_[0]:.2f} | Intercept: {lr.intercept_:.2f} | R²: {r2_lin:.4f}")

    # Polynomial (degree 2): Surge → Demand
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_lin)
    lr_poly = LinearRegression().fit(X_poly, y_dem)
    y_pred_poly = lr_poly.predict(X_poly)
    r2_poly = r2_score(y_dem, y_pred_poly)
    print(f"\n  Polynomial Regression (deg=2): Surge → Demand")
    print(f"     R²: {r2_poly:.4f}")

    # Multiple linear: Surge + DriverSupply + WaitTime → Demand
    X_multi = df[["SurgeMultiplier", "DriverSupply", "WaitTime"]].values
    lr_multi = LinearRegression().fit(X_multi, y_dem)
    r2_multi = r2_score(y_dem, lr_multi.predict(X_multi))
    print(f"\n  Multiple Regression: Surge + DriverSupply + WaitTime → Demand")
    print(f"     Coefficients: Surge={lr_multi.coef_[0]:.2f}, "
          f"DriverSupply={lr_multi.coef_[1]:.2f}, WaitTime={lr_multi.coef_[2]:.2f}")
    print(f"     R²: {r2_multi:.4f}")

    return {
        "lr": lr, "lr_poly": lr_poly, "poly": poly, "lr_multi": lr_multi,
        "r2_lin": r2_lin, "r2_poly": r2_poly, "r2_multi": r2_multi,
        "y_pred_lin": y_pred_lin, "y_pred_poly": y_pred_poly,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_surge_by_timeslot(df):
    """Avg surge multiplier & demand across time slots"""
    set_style()
    order = TIME_SLOTS
    grp = df.groupby("TimeSlot").agg(
        AvgSurge=("SurgeMultiplier","mean"),
        AvgDemand=("Demand","mean"),
    ).reindex(order)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Surge Pricing & Demand Patterns Across Time Slots", fontsize=14, fontweight="bold", color=TEXT)

    bars = axes[0].bar(grp.index, grp["AvgSurge"], color=[CORAL if s > 1.5 else TEAL for s in grp["AvgSurge"]],
                       edgecolor="none", alpha=0.9)
    axes[0].axhline(1.0, color=MUTED, lw=1.2, linestyle="--", label="Baseline (1×)")
    axes[0].set_ylabel("Avg Surge Multiplier")
    axes[0].set_title("Average Surge Multiplier by Time Slot")
    axes[0].legend()
    for bar, v in zip(bars, grp["AvgSurge"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}×",
                     ha="center", va="bottom", fontsize=8.5, color=TEXT)

    axes[1].bar(grp.index, grp["AvgDemand"], color=BLUE, edgecolor="none", alpha=0.9)
    axes[1].set_ylabel("Avg Ride Demand")
    axes[1].set_title("Average Ride Demand by Time Slot")
    axes[1].tick_params(axis="x", rotation=35)
    for spine in ["top","right"]:
        axes[0].spines[spine].set_visible(False)
        axes[1].spines[spine].set_visible(False)

    plt.tight_layout()
    save_fig("surge_by_timeslot")


def plot_pricing_elasticity(df, reg):
    """Demand vs Surge with linear + polynomial regression fits"""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Pricing Elasticity — Demand Response to Surge", fontsize=14, fontweight="bold", color=TEXT)

    # Scatter + regression
    sample = df.sample(1200, random_state=42)
    axes[0].scatter(sample["SurgeMultiplier"], sample["Demand"],
                    alpha=0.25, s=12, color=TEAL, edgecolors="none", label="Observations")

    x_range = np.linspace(df["SurgeMultiplier"].min(), df["SurgeMultiplier"].max(), 200).reshape(-1,1)
    axes[0].plot(x_range, reg["lr"].predict(x_range), color=CORAL, lw=2.2,
                 label=f"Linear (R²={reg['r2_lin']:.3f})")
    x_poly = reg["poly"].transform(x_range)
    axes[0].plot(x_range, reg["lr_poly"].predict(x_poly), color=AMBER, lw=2.2, linestyle="--",
                 label=f"Polynomial deg=2 (R²={reg['r2_poly']:.3f})")
    axes[0].set_xlabel("Surge Multiplier")
    axes[0].set_ylabel("Ride Demand")
    axes[0].set_title("Demand vs Surge — Regression Fit")
    axes[0].legend(fontsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Elasticity bins
    bins    = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.5]
    labels  = ["1.0–1.25×","1.25–1.5×","1.5–1.75×","1.75–2.0×","2.0–2.5×","2.5×+"]
    df2     = df.copy()
    df2["SurgeBin"] = pd.cut(df2["SurgeMultiplier"], bins=bins, labels=labels)
    avg_demand = df2.groupby("SurgeBin", observed=True)["Demand"].mean()

    axes[1].bar(avg_demand.index, avg_demand.values,
                color=[TEAL, TEAL, AMBER, AMBER, CORAL, CORAL], edgecolor="none", alpha=0.9)
    axes[1].set_xlabel("Surge Multiplier Bracket")
    axes[1].set_ylabel("Avg Ride Demand")
    axes[1].set_title("Avg Demand per Surge Bracket")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for i, (label, val) in enumerate(zip(avg_demand.index, avg_demand.values)):
        axes[1].text(i, val + 3, f"{val:.0f}", ha="center", fontsize=9, color=TEXT)

    plt.tight_layout()
    save_fig("pricing_elasticity")


def plot_weather_impact(df):
    """Weather conditions → surge and demand"""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Weather Impact on Surge Pricing & Demand", fontsize=14, fontweight="bold", color=TEXT)

    order   = ["Clear", "Cloudy", "Rain", "Heavy Rain"]
    colors  = [TEAL, BLUE, AMBER, CORAL]
    grp_w   = df.groupby("Weather").agg(AvgSurge=("SurgeMultiplier","mean"),
                                         AvgDemand=("Demand","mean")).reindex(order)

    axes[0].bar(grp_w.index, grp_w["AvgSurge"], color=colors, edgecolor="none", alpha=0.9)
    axes[0].set_ylabel("Avg Surge Multiplier")
    axes[0].set_title("Surge by Weather Condition")
    axes[0].axhline(df["SurgeMultiplier"].mean(), color=MUTED, lw=1.2, linestyle="--", label="Overall avg")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].bar(grp_w.index, grp_w["AvgDemand"], color=colors, edgecolor="none", alpha=0.9)
    axes[1].set_ylabel("Avg Ride Demand")
    axes[1].set_title("Demand by Weather Condition")
    axes[1].axhline(df["Demand"].mean(), color=MUTED, lw=1.2, linestyle="--", label="Overall avg")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig("weather_impact")


def plot_hypothesis_summary(h_results):
    """Visual summary of all hypothesis test results"""
    set_style()
    tests = [
        ("H1", "Rush vs Off-Peak\nDemand (t-test)", h_results["H1_rush_vs_offpeak"]["p"]),
        ("H2", "Rain vs Clear\nSurge (Mann-Whitney)", h_results["H2_rain_vs_clear_surge"]["p"]),
        ("H3", "Weekend vs Weekday\nCancellation (t-test)", h_results["H3_weekend_cancellation"]["p"]),
        ("H4", "Surge vs Rating\n(Pearson)", h_results["H4_surge_rating_corr"]["p"]),
        ("H5", "Surge vs Demand\n(Spearman ρ)", h_results["H5_surge_demand_spearman"]["p"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Hypothesis Testing Summary — p-values vs α=0.05", fontsize=13, fontweight="bold", color=TEXT)

    labels  = [f"{t[0]}: {t[1]}" for t in tests]
    pvals   = [t[2] for t in tests]
    colors  = [TEAL if p < 0.05 else CORAL for p in pvals]
    bars    = ax.barh(labels, pvals, color=colors, edgecolor="none", alpha=0.9, height=0.55)
    ax.axvline(0.05, color=AMBER, lw=2, linestyle="--", label="α = 0.05")
    ax.set_xlabel("p-value")
    ax.set_xlim(0, max(pvals) * 1.3)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for bar, p in zip(bars, pvals):
        label = f"p={p:.4f} {'✅' if p<0.05 else '❌'}"
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=9, color=TEXT)

    plt.tight_layout()
    save_fig("hypothesis_summary")


def plot_correlation_heatmap(df):
    set_style()
    cols = ["SurgeMultiplier","FinalFare","Demand","CancellationRate",
            "DriverSupply","WaitTime","UserRating","IsWeekend"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 9}, linewidths=0.5, linecolor=BORDER,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Ride-Hailing Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("correlation_heatmap")


def plot_city_comparison(df):
    set_style()
    grp = df.groupby("City").agg(
        AvgSurge=("SurgeMultiplier","mean"),
        AvgFare=("FinalFare","mean"),
        AvgDemand=("Demand","mean"),
        AvgCancel=("CancellationRate","mean"),
    ).sort_values("AvgSurge", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("City-wise Comparison — Dynamic Pricing Metrics", fontsize=14, fontweight="bold", color=TEXT)

    metrics = [("AvgSurge","Avg Surge Multiplier","×"), ("AvgFare","Avg Final Fare","₹"),
               ("AvgDemand","Avg Demand","rides"), ("AvgCancel","Avg Cancellation Rate","")]
    for ax, (col, title, unit) in zip(axes.flat, metrics):
        vals   = grp[col]
        colors = [PALETTE[i] for i in range(len(vals))]
        bars   = ax.bar(grp.index, vals, color=colors, edgecolor="none", alpha=0.9)
        ax.set_title(title); ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.01,
                    f"{v:.2f}{unit}", ha="center", fontsize=8, color=TEXT)

    plt.tight_layout()
    save_fig("city_comparison")


def plot_cancellation_analysis(df):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cancellation Rate Analysis", fontsize=14, fontweight="bold", color=TEXT)

    # Scatter: surge vs cancellation
    sample = df.sample(1500, random_state=1)
    axes[0].scatter(sample["SurgeMultiplier"], sample["CancellationRate"],
                    alpha=0.2, s=12, color=CORAL, edgecolors="none")
    m, b = np.polyfit(df["SurgeMultiplier"], df["CancellationRate"], 1)
    x_r  = np.linspace(df["SurgeMultiplier"].min(), df["SurgeMultiplier"].max(), 100)
    axes[0].plot(x_r, m*x_r+b, color=AMBER, lw=2.2, label=f"Trend (slope={m:.3f})")
    axes[0].set_xlabel("Surge Multiplier"); axes[0].set_ylabel("Cancellation Rate")
    axes[0].set_title("Surge vs Cancellation Rate")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    # Cancellation by time slot
    grp_c = df.groupby("TimeSlot")["CancellationRate"].mean().reindex(TIME_SLOTS)
    axes[1].bar(grp_c.index, grp_c.values, color=[CORAL if v > grp_c.mean() else TEAL for v in grp_c.values],
                edgecolor="none", alpha=0.9)
    axes[1].axhline(grp_c.mean(), color=AMBER, lw=1.5, linestyle="--", label="Overall avg")
    axes[1].set_xlabel("Time Slot"); axes[1].set_ylabel("Avg Cancellation Rate")
    axes[1].set_title("Cancellation Rate by Time Slot")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig("cancellation_analysis")


def plot_weekly_heatmap(df):
    """Heatmap of avg surge: Day × Time Slot"""
    set_style()
    pivot = df.pivot_table(values="SurgeMultiplier", index="Day", columns="TimeSlot",
                           aggfunc="mean").reindex(DAYS)[TIME_SLOTS]
    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, annot=True, fmt=".2f", annot_kws={"size": 8.5},
                linewidths=0.4, linecolor=BORDER, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Weekly Surge Heatmap — Day × Time Slot", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Slot"); ax.set_ylabel("Day of Week")
    plt.tight_layout()
    save_fig("weekly_surge_heatmap")


def plot_fare_distribution(df):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fare Distribution Analysis", fontsize=14, fontweight="bold", color=TEXT)

    axes[0].hist(df["FinalFare"], bins=55, color=TEAL, edgecolor="none", alpha=0.85)
    axes[0].axvline(df["FinalFare"].mean(), color=CORAL, lw=2, linestyle="--",
                    label=f"Mean: ₹{df['FinalFare'].mean():.0f}")
    axes[0].axvline(df["FinalFare"].median(), color=AMBER, lw=2, linestyle="--",
                    label=f"Median: ₹{df['FinalFare'].median():.0f}")
    axes[0].set_xlabel("Final Fare (₹)"); axes[0].set_ylabel("Frequency")
    axes[0].set_title("Overall Fare Distribution"); axes[0].legend()
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    # Fare by weather
    data_w = [df[df["Weather"]==w]["FinalFare"].values for w in ["Clear","Cloudy","Rain","Heavy Rain"]]
    bp = axes[1].boxplot(data_w, patch_artist=True, notch=False,
                          medianprops={"color": BG, "linewidth": 2},
                          flierprops={"marker":"o","markerfacecolor":MUTED,"markersize":3,"alpha":0.4})
    for patch, color in zip(bp["boxes"], [TEAL, BLUE, AMBER, CORAL]):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    axes[1].set_xticklabels(["Clear","Cloudy","Rain","Heavy Rain"])
    axes[1].set_xlabel("Weather Condition"); axes[1].set_ylabel("Final Fare (₹)")
    axes[1].set_title("Fare Distribution by Weather")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig("fare_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Statistical Analysis of Dynamic Pricing — Ride-Hailing")
    print("=" * 60)

    print("\n[1/5] Generating synthetic ride dataset …")
    df = generate_dataset(5000)
    df.to_csv("ride_data.csv", index=False)
    print(f"      {df.shape[0]:,} rides | {df.shape[1]} features")

    print("\n[2/5] Descriptive statistics …")
    descriptive_stats(df)

    print("\n[3/5] Hypothesis testing …")
    h_results = hypothesis_tests(df)

    print("\n[4/5] Regression analysis …")
    reg = regression_analysis(df)

    print("\n[5/5] Generating visualisations …")
    plot_surge_by_timeslot(df)
    plot_pricing_elasticity(df, reg)
    plot_weather_impact(df)
    plot_hypothesis_summary(h_results)
    plot_correlation_heatmap(df)
    plot_city_comparison(df)
    plot_cancellation_analysis(df)
    plot_weekly_heatmap(df)
    plot_fare_distribution(df)

    print("\Analysis complete — 9 charts generated.")
