import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from itertools import combinations
from scipy import stats

_DIR      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_DIR, "cache")

st.set_page_config(page_title="Circulation Plan Analysis", layout="wide")
st.title("📊 Circulation Plan Analysis")
st.markdown(
    "<p style='color:grey;font-size:16px;'>"
    "Comparing weather-normalised predicted cyclist counts against observed counts "
    "before and after urban circulation plan changes in Aalst and Kortrijk."
    "</p>",
    unsafe_allow_html=True,
)

AALST_INTERVENTION = pd.Timestamp("2021-08-16")
KORTRIJK_TRIAL     = pd.Timestamp("2022-07-01")
KORTRIJK_PERMANENT = pd.Timestamp("2023-10-01")

CITY_CONFIG = {
    "Aalst": {
        "sites":         {10: "Aalst 1", 11: "Aalst 2", 19: "Aalst 3"},
        "city_site_ids": [10, 11, 19],
        "pre_cutoff":    AALST_INTERVENTION,
        "interventions": [
            {"date": AALST_INTERVENTION, "label": "Circulation plan — Aug 16, 2021",
             "dash": "solid", "color": "crimson"},
        ],
        "note": None,
    },
    "Kortrijk": {
        "sites":         {16: "Kortrijk 1", 17: "Kortrijk 2"},
        "city_site_ids": [16, 17],
        "pre_cutoff":    KORTRIJK_TRIAL,
        "interventions": [
            {"date": KORTRIJK_TRIAL,     "label": "Trial phase — Summer 2022",
             "dash": "dash",  "color": "orange"},
            {"date": KORTRIJK_PERMANENT, "label": "Permanent plan — Oct 2023",
             "dash": "solid", "color": "crimson"},
        ],
        "note": (
            "ℹ️ **About the Kortrijk phases**  \n"
            "**Before (Aug 2019 – Jun 2022):** baseline period before any intervention.  \n"
            "**Trial phase (Jul 2022 – Sep 2023):** temporary bollards and signage redirected "
            "through-traffic away from the city centre, creating a low-traffic neighbourhood.  \n"
            "**Permanent plan (Oct 2023 onwards):** physical infrastructure changes made the "
            "plan permanent."
        ),
    },
}

missing = [
    city for city in CITY_CONFIG
    if not os.path.exists(os.path.join(CACHE_DIR, f"{city.lower()}_daily.parquet"))
]
if missing:
    st.error(
        f"**Cache files not found for: {', '.join(missing)}.**\n\n"
        "Run the preparation script once from your terminal to generate them:\n\n"
        "```\ncd weather_model\npython prepare_case_study.py\n```\n\n"
        "This trains the city-specific models and pre-computes all predictions. "
        "It only needs to run again if the source data changes."
    )
    st.stop()

@st.cache_data(show_spinner="Loading predictions…")
def load_city_daily(city):
    path = os.path.join(CACHE_DIR, f"{city.lower()}_daily.parquet")
    df = pd.read_parquet(path)
    df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"])
    return df

def fmt_p(p):
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"

def dunn_posthoc(groups, labels):
    """Dunn's post-hoc test with Holm–Bonferroni correction."""
    all_data  = np.concatenate(groups)
    N         = len(all_data)
    all_ranks = stats.rankdata(all_data)

    mean_ranks, sizes = [], []
    idx = 0
    for g in groups:
        n = len(g)
        mean_ranks.append(all_ranks[idx : idx + n].mean())
        sizes.append(n)
        idx += n

    rows = []
    for (i, j) in combinations(range(len(groups)), 2):
        se    = np.sqrt((N * (N + 1) / 12.0) * (1.0 / sizes[i] + 1.0 / sizes[j]))
        z     = (mean_ranks[i] - mean_ranks[j]) / se
        p_raw = float(2 * stats.norm.sf(abs(z)))
        rows.append({"Group A": labels[i], "Group B": labels[j], "z": z, "p_raw": p_raw})

    df = pd.DataFrame(rows)

    # Holm–Bonferroni step-down correction
    m       = len(df)
    order   = df["p_raw"].argsort().values
    p_adj   = np.empty(m)
    running = 0.0
    for rank, orig_idx in enumerate(order):
        corrected = df.loc[orig_idx, "p_raw"] * (m - rank)
        running   = max(running, corrected)  # step-down monotonicity
        p_adj[orig_idx] = min(1.0, running)

    df["p_adj"] = p_adj
    return df

city = st.sidebar.radio("City", list(CITY_CONFIG.keys()))
cfg  = CITY_CONFIG[city]

if cfg["note"]:
    st.info(cfg["note"])

selected_ids = st.sidebar.multiselect(
    "Monitoring sites",
    options=list(cfg["sites"].keys()),
    default=list(cfg["sites"].keys()),
    format_func=lambda x: cfg["sites"][x],
)
smoothing = st.sidebar.selectbox(
    "Smoothing", ["Daily", "Weekly (7d)", "Monthly (30d)"], index=1
)
window = {"Daily": 1, "Weekly (7d)": 7, "Monthly (30d)": 30}[smoothing]

if not selected_ids:
    st.warning("Select at least one monitoring site.")
    st.stop()

city_daily = load_city_daily(city)

st.markdown("---")

for site_id in selected_ids:
    site_name = cfg["sites"][site_id]

    daily = (
        city_daily[city_daily["site_id"] == site_id]
        .set_index("hour_timestamp")
        [["bike_count", "predicted"]]
        .sort_index()
    )
    daily["residual"] = daily["bike_count"] - daily["predicted"]

    # 90% prediction interval from pre-intervention residuals
    pre_mask  = daily.index < cfg["pre_cutoff"]
    pre_resid = daily.loc[pre_mask, "residual"]
    if len(pre_resid) >= 10:
        lo_band = daily["predicted"] + pre_resid.quantile(0.05)
        hi_band = daily["predicted"] + pre_resid.quantile(0.95)
    else:
        lo_band = hi_band = None  # Kortrijk: no pre-intervention data

    sm = daily[["bike_count", "predicted"]].rolling(window, min_periods=1, center=True).mean()
    if lo_band is not None:
        lo_s = lo_band.rolling(window, min_periods=1, center=True).mean()
        hi_s = hi_band.rolling(window, min_periods=1, center=True).mean()

    fig = go.Figure()

    if lo_band is not None:
        fig.add_trace(go.Scatter(
            x=pd.concat([daily.index.to_series(), daily.index.to_series()[::-1]]),
            y=pd.concat([hi_s, lo_s[::-1]]),
            fill="toself",
            fillcolor="rgba(255,127,14,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% prediction interval",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=sm.index, y=sm["bike_count"],
        name="Actual", line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=sm.index, y=sm["predicted"],
        name="Predicted (weather baseline)",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))

    for iv in cfg["interventions"]:
        fig.add_vline(
            x=iv["date"].timestamp() * 1000,
            line_dash=iv["dash"], line_color=iv["color"], line_width=2,
            annotation_text=iv["label"],
            annotation_position="top left", annotation_font_size=11,
        )

    fig.update_layout(
        title=f"{site_name} — Actual vs Weather-Normalised Baseline",
        xaxis_title="Date",
        yaxis_title=f"Cyclist count ({smoothing.lower()} avg)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=420,
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    weekly = daily["residual"].resample("W").mean()

    if city == "Aalst":
        cutoff = AALST_INTERVENTION
        post = daily.loc[daily.index >= cutoff]
        pre  = daily.loc[daily.index <  cutoff]

        actual_total    = int(post["bike_count"].sum())
        predicted_total = int(post["predicted"].sum())
        div_pct = (actual_total - predicted_total) / predicted_total * 100 if predicted_total else 0

        avg_actual_pre  = pre["bike_count"].mean()
        avg_actual_post = post["bike_count"].mean()
        avg_daily_delta = avg_actual_post - avg_actual_pre
        avg_daily_pct   = avg_daily_delta / avg_actual_pre * 100 if avg_actual_pre else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total actual cyclists (post)",    f"{actual_total:,}")
        c2.metric("Total predicted baseline (post)", f"{predicted_total:,}")
        c3.metric(
            "Post-intervention vs baseline",
            f"{div_pct:+.1f}%",
            delta_color="normal" if div_pct >= 0 else "inverse",
        )

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg daily cyclists — before", f"{avg_actual_pre:,.0f}")
        c5.metric(
            "Avg daily cyclists — after",
            f"{avg_actual_post:,.0f}",
            delta=f"{avg_daily_delta:+.0f}/day vs before",
        )
        c6.metric(
            "Change before → after",
            f"{avg_daily_pct:+.1f}%",
            delta_color="normal" if avg_daily_pct >= 0 else "inverse",
        )

        pre_w  = weekly[weekly.index < cutoff].dropna()
        post_w = weekly[weekly.index >= cutoff].dropna()
        if len(pre_w) >= 5 and len(post_w) >= 5:
            _, p = stats.mannwhitneyu(pre_w, post_w, alternative="two-sided")
            p_label = fmt_p(p)
            if p < 0.05:
                st.success(f"✅ Statistically significant change detected (Wilcoxon rank-sum, {p_label})")
            else:
                st.info(f"ℹ️ No statistically significant change detected (Wilcoxon rank-sum, {p_label})")

        with st.expander("ℹ️ How is this measured?"):
            st.markdown(
                "An XGBoost model trained on pre-intervention data gives a weather-adjusted baseline — "
                "what counts would have looked like without the circulation plan. "
                "The residuals (actual minus predicted) remove the weather effect. "
                "A Wilcoxon rank-sum test on weekly averages checks whether the shift is significant; "
                "weekly aggregation avoids inflating the sample size with day-to-day autocorrelation."
            )

    else:  # Kortrijk — 3 phases
        pre   = daily.loc[daily.index < KORTRIJK_TRIAL]
        trial = daily.loc[(daily.index >= KORTRIJK_TRIAL) & (daily.index < KORTRIJK_PERMANENT)]
        perm  = daily.loc[daily.index >= KORTRIJK_PERMANENT]

        pre_avg   = pre["bike_count"].mean()
        trial_avg = trial["bike_count"].mean()
        perm_avg  = perm["bike_count"].mean()

        trial_delta = trial_avg - pre_avg
        trial_pct   = trial_delta / pre_avg * 100 if pre_avg else 0
        perm_delta  = perm_avg - pre_avg
        perm_pct    = perm_delta / pre_avg * 100 if pre_avg else 0

        trial_pred_avg    = trial["predicted"].mean()
        perm_pred_avg     = perm["predicted"].mean()
        trial_vs_baseline = (trial_avg - trial_pred_avg) / trial_pred_avg * 100 if trial_pred_avg else 0
        perm_vs_baseline  = (perm_avg  - perm_pred_avg)  / perm_pred_avg  * 100 if perm_pred_avg  else 0
        trial_resid       = trial["residual"].mean()
        perm_resid        = perm["residual"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg daily cyclists — before",    f"{pre_avg:,.0f}")
        c2.metric("Avg daily cyclists — trial",     f"{trial_avg:,.0f}", delta=f"{trial_delta:+.0f}/day vs before")
        c3.metric("Avg daily cyclists — permanent", f"{perm_avg:,.0f}",  delta=f"{perm_delta:+.0f}/day vs before")

        trial_to_perm_delta = perm_avg - trial_avg
        trial_to_perm_pct   = trial_to_perm_delta / trial_avg * 100 if trial_avg else 0

        c4, c5, c6 = st.columns(3)
        c4.metric(
            "Before → trial",
            f"{trial_pct:+.1f}%",
            delta=f"{trial_delta:+.0f} cyclists/day",
            delta_color="normal" if trial_pct >= 0 else "inverse",
        )
        c5.metric(
            "Trial → permanent",
            f"{trial_to_perm_pct:+.1f}%",
            delta=f"{trial_to_perm_delta:+.0f} cyclists/day",
            delta_color="normal" if trial_to_perm_pct >= 0 else "inverse",
        )
        c6.metric(
            "Before → permanent (total)",
            f"{perm_pct:+.1f}%",
            delta=f"{perm_delta:+.0f} cyclists/day",
            delta_color="normal" if perm_pct >= 0 else "inverse",
        )

        # Kruskal-Wallis omnibus → Dunn's post-hoc
        pre_w   = weekly[weekly.index < KORTRIJK_TRIAL].dropna()
        trial_w = weekly[(weekly.index >= KORTRIJK_TRIAL) & (weekly.index < KORTRIJK_PERMANENT)].dropna()
        perm_w  = weekly[weekly.index >= KORTRIJK_PERMANENT].dropna()

        enough = len(pre_w) >= 5 and len(trial_w) >= 5 and len(perm_w) >= 5
        if enough:
            stat_kw, p_kw = stats.kruskal(pre_w, trial_w, perm_w)
            kw_label = fmt_p(p_kw)

            if p_kw >= 0.05:
                st.info(
                    f"ℹ️ Kruskal-Wallis found no significant difference across the three phases "
                    f"({kw_label}). Post-hoc tests not warranted."
                )
            else:
                st.success(f"✅ Kruskal-Wallis: at least one phase differs significantly (H = {stat_kw:.2f}, {kw_label})")

                dunn = dunn_posthoc(
                    [pre_w.values, trial_w.values, perm_w.values],
                    ["Before", "Trial", "Permanent"],
                )
                for _, row in dunn.iterrows():
                    label   = f"{row['Group A']} vs {row['Group B']}"
                    p_raw   = fmt_p(row["p_raw"])
                    p_adj   = fmt_p(row["p_adj"])
                    sig     = row["p_adj"] < 0.05
                    fn      = st.success if sig else st.info
                    outcome = "significant" if sig else "not significant"
                    fn(f"{'✅' if sig else 'ℹ️'} **{label}**: {outcome} after correction  (p_raw {p_raw} → p_adj {p_adj}, Holm–Bonferroni)")
        else:
            st.warning("⚠️ Not enough data in one or more phases for significance testing (need ≥ 5 weeks each).")

        with st.expander("ℹ️ How is this measured?"):
            st.markdown(
                "Three periods are compared: before (Aug 2019 – Jun 2022), trial (Jul 2022 – Sep 2023), "
                "and permanent (Oct 2023 onwards). The XGBoost baseline controls for seasonal differences "
                "between periods. We first run a Kruskal-Wallis test to check whether any phase differs, "
                "then Dunn's post-hoc with Holm–Bonferroni correction for the three pairwise comparisons. "
                "Weekly aggregation keeps autocorrelation from inflating the sample size."
            )

    st.markdown("---")
