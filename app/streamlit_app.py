"""
Streamlit dashboard.

Five tabs, each answering a different question:

  Forecast    What will the air be like? (the default view)
  Accuracy    How much should I trust that? (backtest curves, skill vs baselines)
  Explain     What is driving it? (SHAP, global and grouped)
  Explore     What does the history look like? (EDA)
  System      Is the pipeline healthy? (store, registry, leaderboard)

Reads the model and feature store directly rather than going through the API, so
the dashboard still works when the API is not running. `app/api.py` serves the
same data for programmatic consumers.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import alerts, config, evaluate, feature_store, model_registry, predict  # noqa: E402

st.set_page_config(
    page_title=f"AQI Forecast - {config.CITY_NAME}",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Band colours reused across every chart so a colour always means the same thing.
BAND_COLORS = [
    (0, 50, "#22c55e", "Good"),
    (50, 100, "#eab308", "Moderate"),
    (100, 150, "#f97316", "Unhealthy (Sensitive)"),
    (150, 200, "#ef4444", "Unhealthy"),
    (200, 300, "#a855f7", "Very Unhealthy"),
    (300, 500, "#7f1d1d", "Hazardous"),
]

st.markdown("""
<style>
  .metric-card {
      border-radius: 14px; padding: 18px 20px; color: #fff;
      box-shadow: 0 2px 10px rgba(0,0,0,.14);
  }
  .metric-card h2 { margin: 0; font-size: 2.6rem; line-height: 1.1; }
  .metric-card p  { margin: 2px 0 0; opacity: .92; font-size: .92rem; }
  .band-pill {
      display:inline-block; padding:3px 11px; border-radius:999px;
      font-size:.78rem; color:#fff; font-weight:600;
  }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 10px 18px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner="Generating forecast...")
def load_forecast() -> pd.DataFrame:
    return predict.forecast()


@st.cache_data(ttl=900)
def load_current() -> dict:
    return predict.current_conditions()


@st.cache_data(ttl=3600)
def load_history(days: int = 120) -> pd.DataFrame:
    raw = feature_store.read_observations()
    return raw.tail(days * 24)


@st.cache_data(ttl=3600)
def load_horizon_metrics() -> pd.DataFrame | None:
    if not config.HORIZON_METRICS_PATH.exists():
        return None
    return pd.read_parquet(config.HORIZON_METRICS_PATH)


@st.cache_data(ttl=3600)
def load_model_comparison() -> pd.DataFrame | None:
    path = config.DATA_DIR / "model_comparison.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600, show_spinner="Computing SHAP values...")
def load_explanation():
    from src import explainability

    return explainability.explain_production_model()


def add_band_shading(fig, x0, x1, row=None, col=None) -> None:
    """Paint the EPA category bands behind a chart."""
    for low, high, colour, _ in BAND_COLORS:
        kwargs = {"row": row, "col": col} if row else {}
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=low, y1=high,
                      fillcolor=colour, opacity=0.10, layer="below",
                      line_width=0, **kwargs)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🌫️ AQI Predictor")
    st.caption(f"{config.CITY_NAME} · {config.LATITUDE:.3f}, {config.LONGITUDE:.3f}")

    try:
        store_info = feature_store.describe()
        entry = model_registry.production_entry()
    except Exception:  # noqa: BLE001
        store_info, entry = {"available": False}, None

    if store_info.get("available") and entry:
        st.success("Pipeline ready")
    else:
        st.error("Pipeline not ready")
        st.code("python -m src.backfill\npython -m src.train", language="bash")
        st.stop()

    st.metric("Observations", f"{store_info['rows']:,}")
    st.caption(f"History to {pd.Timestamp(store_info['end']).strftime('%d %b %Y %H:%M')}")

    st.divider()
    st.subheader("Production model")
    st.write(f"**{entry.get('model_name', 'n/a')}**")
    metrics = entry.get("metrics", {})
    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{metrics.get('rmse', float('nan')):.1f}")
    c2.metric("R²", f"{metrics.get('r2', float('nan')):.3f}")
    if "skill_vs_persistence" in metrics:
        st.metric("Skill vs persistence", f"{metrics['skill_vs_persistence'] * 100:+.1f}%",
                  help="RMSE reduction against simply repeating the last observed AQI. "
                       "Positive means the model genuinely adds information.")

    st.divider()
    alert_threshold = st.slider("Alert threshold (AQI)", 50, 300,
                                int(config.ALERT_AQI_THRESHOLD), step=10)
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
try:
    current = load_current()
    forecast_df = load_forecast()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not produce a forecast: {exc}")
    st.stop()

daily = predict.daily_summary(forecast_df)
alert_payload = alerts.build_alert(forecast_df, threshold=float(alert_threshold))

st.title(f"Air Quality Forecast · {config.CITY_NAME}")
st.caption(
    f"Forecast origin {pd.Timestamp(forecast_df['forecast_origin'].iloc[0]).strftime('%d %b %Y %H:%M')} "
    f"· next {config.FORECAST_HORIZON_DAYS} days · model {entry.get('model_name')}"
)

if alert_payload["alert"]:
    st.warning(f"**{alert_payload['severity'].upper()}** — {alert_payload['message']}\n\n"
               f"{alert_payload['advice']}", icon="⚠️")

cols = st.columns([1.4, 1, 1, 1])
with cols[0]:
    # Label the observation by its own timestamp, and say so when the feature
    # pipeline has not run recently rather than passing stale data off as "now".
    observed_label = ("LATEST OBSERVED" if current.get("stale") else "NOW")
    age_note = (f" · {current['age_hours']:.0f}h ago" if current.get("stale") else "")
    st.markdown(
        f"""<div class="metric-card" style="background:{current['color']}">
              <p>{observed_label} · {pd.Timestamp(current['timestamp']).strftime('%H:%M')}{age_note}</p>
              <h2>{current['aqi']:.0f}</h2>
              <p>{current['category']}</p>
            </div>""", unsafe_allow_html=True)

for i, (_, row) in enumerate(daily.head(3).iterrows()):
    label = ["Today", "Tomorrow", "Day 3"][i] if i < 3 else str(row["date"])
    with cols[i + 1]:
        st.markdown(
            f"""<div class="metric-card" style="background:{row['color']}">
                  <p>{label.upper()} · {pd.Timestamp(row['date']).strftime('%a %d %b')}</p>
                  <h2>{row['avg_aqi']:.0f}</h2>
                  <p>{row['min_aqi']:.0f}–{row['max_aqi']:.0f} · peak {row['peak_hour']}</p>
                </div>""", unsafe_allow_html=True)

st.write("")
tab_forecast, tab_accuracy, tab_explain, tab_explore, tab_system = st.tabs(
    ["📈 Forecast", "🎯 Accuracy", "🔍 Explain", "📊 Explore", "⚙️ System"]
)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
with tab_forecast:
    history = load_history(days=7)
    fig = go.Figure()

    add_band_shading(fig, history.index.min(), forecast_df.index.max())

    fig.add_trace(go.Scatter(
        x=history.index, y=history[config.TARGET_COLUMN],
        name="Observed", mode="lines", line=dict(color="#0f172a", width=2),
    ))

    if forecast_df["upper_80"].notna().any():
        fig.add_trace(go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df["upper_80"]) + list(forecast_df["lower_80"][::-1]),
            fill="toself", fillcolor="rgba(37,99,235,.18)", line=dict(width=0),
            name="80% interval", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["predicted_aqi"],
        name="Forecast", mode="lines", line=dict(color="#2563eb", width=3, dash="solid"),
    ))
    # Plotly wants a string for a shape on a date axis; a bare Timestamp is
    # rejected by several 5.x releases.
    fig.add_vline(x=pd.Timestamp(forecast_df["forecast_origin"].iloc[0]).isoformat(),
                  line_dash="dot", line_color="#64748b", annotation_text="now")

    fig.update_layout(
        height=460, hovermode="x unified", margin=dict(t=30, b=10, l=10, r=10),
        yaxis_title="US AQI", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Day by day")
    for _, row in daily.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([1, 4])
            c1.markdown(
                f"**{pd.Timestamp(row['date']).strftime('%A %d %b')}**<br>"
                f"<span class='band-pill' style='background:{row['color']}'>"
                f"{row['category']}</span>", unsafe_allow_html=True)
            c2.write(
                f"Average **{row['avg_aqi']:.0f}**, ranging {row['min_aqi']:.0f}–"
                f"{row['max_aqi']:.0f}, peaking around **{row['peak_hour']}**."
            )
            c2.caption(row["advice"])

    if alert_payload["episodes"]:
        st.subheader(f"Episodes above AQI {alert_threshold}")
        st.dataframe(pd.DataFrame(alert_payload["episodes"])[
            ["start", "end", "duration_hours", "peak_aqi", "category", "severity"]
        ], use_container_width=True, hide_index=True)

    st.download_button(
        "Download forecast (CSV)",
        forecast_df.reset_index().to_csv(index=False).encode(),
        file_name=f"aqi_forecast_{config.CITY_NAME.lower()}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
with tab_accuracy:
    st.subheader("How accurate is this forecast?")
    st.caption(
        "Every figure below comes from walk-forward backtesting: the model is "
        "trained up to a cutoff, scored on the period that follows, and the cutoff "
        "is rolled forward. No test row was ever seen during training."
    )

    horizon = load_horizon_metrics()
    if horizon is None:
        st.info("No backtest metrics yet. Run `python -m src.train`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        day1 = horizon[horizon["horizon_h"] <= 24]
        day3 = horizon[horizon["horizon_h"] > 48]
        c1.metric("Day-1 RMSE", f"{day1['rmse'].mean():.1f}")
        c2.metric("Day-3 RMSE", f"{day3['rmse'].mean():.1f}")
        c3.metric("Day-1 R²", f"{day1['r2'].mean():.3f}")
        c4.metric("Day-3 R²", f"{day3['r2'].mean():.3f}")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
            subplot_titles=("Error grows with lead time",
                            "Skill against persistence (share of error removed)"),
        )
        fig.add_trace(go.Scatter(x=horizon["horizon_h"], y=horizon["rmse"],
                                 name="Model RMSE", line=dict(color="#2563eb", width=3)),
                      row=1, col=1)
        if "persistence_rmse" in horizon.columns:
            fig.add_trace(go.Scatter(x=horizon["horizon_h"], y=horizon["persistence_rmse"],
                                     name="Persistence RMSE",
                                     line=dict(color="#94a3b8", width=2, dash="dash")),
                          row=1, col=1)
        if "skill_vs_persistence" in horizon.columns:
            fig.add_trace(go.Scatter(x=horizon["horizon_h"], y=horizon["skill_vs_persistence"] * 100,
                                     name="Skill %", fill="tozeroy",
                                     line=dict(color="#16a34a", width=2)),
                          row=2, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="#ef4444", row=2, col=1)

        fig.update_xaxes(title_text="Lead time (hours)", row=2, col=1)
        fig.update_yaxes(title_text="RMSE (AQI)", row=1, col=1)
        fig.update_yaxes(title_text="Skill (%)", row=2, col=1)
        fig.update_layout(height=560, hovermode="x unified",
                          margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        if "category_accuracy" in horizon.columns:
            st.subheader("Right AQI category, by lead time")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=horizon["horizon_h"],
                                      y=horizon["category_accuracy"] * 100,
                                      fill="tozeroy", line=dict(color="#7c3aed", width=2),
                                      name="Exact band"))
            if "within_one_band" in horizon.columns:
                fig2.add_trace(go.Scatter(x=horizon["horizon_h"],
                                          y=horizon["within_one_band"] * 100,
                                          line=dict(color="#a855f7", width=2, dash="dash"),
                                          name="Within one band"))
            fig2.update_layout(height=330, yaxis_title="Accuracy (%)",
                               xaxis_title="Lead time (hours)",
                               margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True)

    comparison = load_model_comparison()
    if comparison is not None:
        st.subheader("Model comparison")
        st.caption("Mean across walk-forward folds. Baselines are scored on identical test rows.")
        show = [c for c in ("model", "rmse", "rmse_std", "mae", "r2",
                            "skill_vs_persistence", "category_accuracy", "n_folds")
                if c in comparison.columns]
        st.dataframe(
            comparison[show].style.format({
                "rmse": "{:.2f}", "rmse_std": "{:.2f}", "mae": "{:.2f}", "r2": "{:.3f}",
                "skill_vs_persistence": "{:+.3f}", "category_accuracy": "{:.3f}",
            }).background_gradient(subset=["rmse"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("What drives the forecast?")
    try:
        importance, groups = load_explanation()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Explanations unavailable: {exc}")
        importance = groups = None

    if importance is not None:
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown("**By driver group**")
            fig = go.Figure(go.Bar(
                x=groups["share"] * 100, y=groups["group"], orientation="h",
                marker_color="#2563eb",
                text=[f"{v * 100:.1f}%" for v in groups["share"]], textposition="auto",
            ))
            fig.update_layout(height=340, xaxis_title="Share of total |SHAP| (%)",
                              yaxis=dict(autorange="reversed"),
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Top individual features**")
            top = importance.head(18).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"],
                                   orientation="h", marker_color="#7c3aed"))
            fig.update_layout(height=520, xaxis_title="Mean |SHAP|",
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        # Describe what the numbers actually say rather than asserting a fixed
        # story: which group leads is a property of the trained model, and it
        # changes between runs.
        shares = dict(zip(groups["group"], groups["share"]))
        forecast_share = shares.get("Forecast weather", 0.0)
        history_share = shares.get("AQI history", 0.0)

        if forecast_share > history_share:
            st.caption(
                f"SHAP attributes each prediction across the features that produced it. "
                f"Forecast weather leads at {forecast_share:.0%}, ahead of AQI history at "
                f"{history_share:.0%} - the model leans hardest on what the atmosphere will "
                f"be doing at the target hour, not on where AQI sits now. That is exactly "
                f"the information a recursive forecaster cannot use, and it is where the "
                f"advantage over persistence comes from."
            )
        else:
            st.caption(
                f"SHAP attributes each prediction across the features that produced it. "
                f"AQI history leads at {history_share:.0%}, which is expected - air quality "
                f"is strongly autocorrelated, and that is why persistence is such a "
                f"demanding baseline. Forecast weather at {forecast_share:.0%} is what lets "
                f"the model depart from persistence and anticipate change."
            )


# ---------------------------------------------------------------------------
# Explore
# ---------------------------------------------------------------------------
with tab_explore:
    st.subheader("Historical patterns")
    window = st.select_slider("History window", options=[30, 60, 120, 365, 1000],
                              value=120, format_func=lambda d: f"{d} days")
    hist = load_history(days=window)
    target = config.TARGET_COLUMN

    fig = go.Figure()
    add_band_shading(fig, hist.index.min(), hist.index.max())
    fig.add_trace(go.Scatter(x=hist.index, y=hist[target], mode="lines",
                             line=dict(color="#0f172a", width=1), name="AQI"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist[target].rolling(24 * 7).mean(),
                             mode="lines", line=dict(color="#f59e0b", width=2.5),
                             name="7-day mean"))
    fig.update_layout(height=380, yaxis_title="US AQI", hovermode="x unified",
                      margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Daily cycle**")
        by_hour = hist.groupby(hist.index.hour)[target].agg(["mean", "std"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(by_hour.index) + list(by_hour.index[::-1]),
            y=list(by_hour["mean"] + by_hour["std"]) + list((by_hour["mean"] - by_hour["std"])[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,.15)", line=dict(width=0),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour["mean"],
                                 line=dict(color="#2563eb", width=3), name="Mean AQI"))
        fig.update_layout(height=320, xaxis_title="Hour of day", yaxis_title="AQI",
                          margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Seasonal cycle**")
        by_month = hist.groupby(hist.index.month)[target].mean()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = go.Figure(go.Bar(
            x=[month_names[m - 1] for m in by_month.index], y=by_month.values,
            marker_color=[predict.aqi_category(v)[1] for v in by_month.values]))
        fig.update_layout(height=320, yaxis_title="Mean AQI",
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**What correlates with AQI**")
    numeric = hist.select_dtypes(include=[np.number])
    corr = numeric.corr()[target].drop(target).sort_values()
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=["#ef4444" if v < 0 else "#2563eb" for v in corr.values]))
    fig.update_layout(height=460, xaxis_title=f"Pearson correlation with {target}",
                      margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Time in each category**")
    bands = pd.Series(evaluate.aqi_band_index(hist[target].to_numpy()))
    counts = bands.value_counts().sort_index()
    dist = pd.DataFrame({
        "Category": [evaluate.AQI_BANDS[i][2] for i in counts.index],
        "Hours": counts.values,
        "Share": (counts.values / counts.sum() * 100).round(1),
    })
    st.dataframe(dist, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
with tab_system:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Feature store")
        st.json(store_info)
    with c2:
        st.subheader("Production model")
        st.json({
            "model_name": entry.get("model_name"),
            "trained_at": entry.get("trained_at"),
            "promoted_at": entry.get("promoted_at"),
            "n_features": entry.get("n_features"),
            "selection": entry.get("selection"),
            "data_fingerprint": entry.get("data_fingerprint"),
        })

    st.subheader("Training history")
    board = model_registry.leaderboard(limit=15)
    if board:
        st.dataframe(pd.DataFrame([{
            "model": e.get("model_name"),
            "trained_at": e.get("trained_at"),
            "rmse": e.get("metrics", {}).get("rmse"),
            "mae": e.get("metrics", {}).get("mae"),
            "r2": e.get("metrics", {}).get("r2"),
            "production": bool(e.get("is_production")),
        } for e in board]), use_container_width=True, hide_index=True)

    st.subheader("Pipeline commands")
    st.code(
        "python -m src.backfill           # rebuild the historical dataset\n"
        "python -m src.feature_pipeline   # hourly refresh\n"
        "python -m src.train              # retrain + promote\n"
        "python -m src.report             # regenerate the written report\n"
        "uvicorn app.api:app --port 8000  # REST API",
        language="bash",
    )
