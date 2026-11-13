"""Browser-deployable 1-72 h AQI forecaster for Lahore (direct, horizon-conditioned global model).

Inputs at forecast origin t  : AQI lags, 24 h rolling stats, PM2.5/PM10/NO2/CO at t
Inputs at target time t + h  : forecast weather (temperature, humidity, wind, rain, pressure, cloud)
                               + calendar (hour, day-of-year) + horizon h
Outputs                      : median forecast + 10 % / 90 % quantiles (80 % interval)
Validation                   : chronological hold-out (last 20 %), compared with persistence and
                               24 h seasonal-naive baselines. Exported to ONNX for the web demo.
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web_demo")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

t = pq.read_table(os.path.join(HERE, "..", "data", "raw_observations.parquet")).to_pandas()
t = (t.set_index("timestamp") if "timestamp" in t.columns else t).sort_index()
t = t[~t.index.duplicated()].asfreq("h")
t = t.interpolate(limit=6)
a = t["aqi"].values.astype(float)
n = len(t)

LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 72]
WX = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "surface_pressure", "cloud_cover"]
POLL = ["pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide"]
FEATURES = ([f"aqi_lag{l}" for l in LAGS] + ["aqi_mean24", "aqi_std24", "aqi_min24", "aqi_max24", "aqi_delta3"] + POLL
            + [f"{w}_target" for w in WX] + ["wind_dir_sin_target", "wind_dir_cos_target",
               "hour_sin", "hour_cos", "doy_sin", "doy_cos", "horizon"])

roll = t["aqi"].rolling(24, min_periods=12)
mean24, std24, min24, max24 = (roll.mean().values, roll.std().values, roll.min().values, roll.max().values)
hours = t.index.hour.values; doy = t.index.dayofyear.values
wd = np.deg2rad(t["wind_direction_10m"].values)


def row(i, h):
    j = i + h
    f = [a[i - l] for l in LAGS] + [mean24[i], std24[i], min24[i], max24[i], a[i] - a[i - 3]]
    f += [t[p].values[i] for p in POLL]
    f += [t[w].values[j] for w in WX] + [np.sin(wd[j]), np.cos(wd[j])]
    f += [np.sin(2 * np.pi * hours[j] / 24), np.cos(2 * np.pi * hours[j] / 24),
          np.sin(2 * np.pi * doy[j] / 365.25), np.cos(2 * np.pi * doy[j] / 365.25), h]
    return f


X, Y, I, H = [], [], [], []
for i in range(72, n - 72):
    for h in rng.choice(np.arange(1, 73), 6, replace=False):
        if np.isnan(a[i + h]):
            continue
        X.append(row(i, h)); Y.append(a[i + h]); I.append(i); H.append(h)
X, Y, I, H = map(np.asarray, (X, Y, I, H))
ok = ~np.isnan(X).any(1); X, Y, I, H = X[ok], Y[ok], I[ok], H[ok]
cut = int(np.quantile(I, 0.8))
tr, te = I < cut - 72, I >= cut  # 72 h gap so targets never overlap
print("rows", len(X), "train", tr.sum(), "test", te.sum(), "test starts", t.index[cut])

cal_cut = int(np.quantile(I[tr], 0.85))
fit, cal = I < cal_cut - 72, (I >= cal_cut) & tr
print("fit", fit.sum(), "calibration", cal.sum())

kw = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=40, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
med = HistGradientBoostingRegressor(loss="absolute_error", **kw).fit(X[fit], Y[fit])
q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **kw).fit(X[fit], Y[fit])
q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **kw).fit(X[fit], Y[fit])

# Conformalized quantile regression per horizon bucket -> calibrated 80 % interval
BUCKETS = [(1, 6), (7, 24), (25, 48), (49, 72)]
lc, hc, yc, hcb = q10.predict(X[cal]), q90.predict(X[cal]), Y[cal], H[cal]
offsets = []
for h0, h1 in BUCKETS:
    s = (hcb >= h0) & (hcb <= h1)
    sc = np.maximum(lc[s] - yc[s], yc[s] - hc[s])
    offsets.append(float(np.quantile(sc, 0.8 * (1 + 1 / s.sum()))))
print("conformal offsets", offsets)


def off(h):
    return np.select([(h >= b0) & (h <= b1) for b0, b1 in BUCKETS], offsets)


p, yt, ht = med.predict(X[te]), Y[te], H[te]
lo, hi = q10.predict(X[te]) - off(ht), q90.predict(X[te]) + off(ht)
persist = X[te][:, 0]
seasonal = np.array([a[i + h - 24 * int(np.ceil(h / 24))] for i, h in zip(I[te], ht)])


def m(y, f):
    e = f - y
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


band = lambda v: np.digitize(v, [50, 100, 150, 200, 300])
res = {"overall": m(yt, p), "persistence": m(yt, persist), "seasonal_naive_24h": m(yt, seasonal),
       "interval_coverage_80": float(np.mean((yt >= lo) & (yt <= hi))),
       "category_accuracy": float(np.mean(band(yt) == band(p))),
       "within_one_band": float(np.mean(np.abs(band(yt) - band(p)) <= 1)), "by_horizon": []}
for h0, h1 in BUCKETS:
    s = (ht >= h0) & (ht <= h1)
    res["by_horizon"].append({"range": f"{h0}-{h1}h", **m(yt[s], p[s]), "persistence_rmse": m(yt[s], persist[s])["rmse"],
                              "seasonal_rmse": m(yt[s], seasonal[s])["rmse"], "coverage_80": float(np.mean((yt[s] >= lo[s]) & (yt[s] <= hi[s])))})
res["test_period"] = [str(t.index[cut]), str(t.index[-73])]
res["train_rows"], res["calibration_rows"], res["test_rows"] = int(fit.sum()), int(cal.sum()), int(te.sum())
imp_idx = np.argsort(-np.abs(np.corrcoef(np.c_[X[te][:3000], p[:3000]].T)[-1, :-1]))[:8]
print(json.dumps(res, indent=1))

# sample backtest trace (h=24) for the dashboard
s24 = np.where(ht == 24)[0][:: max(1, len(np.where(ht == 24)[0]) // 300)]
trace = [{"t": str(t.index[I[te][k] + 24]), "y": round(float(yt[k]), 1), "p": round(float(p[k]), 1),
          "lo": round(float(lo[k]), 1), "hi": round(float(hi[k]), 1)} for k in sorted(s24, key=lambda k: I[te][k])]


def export(mdl):
    trees = []
    for it in mdl._predictors:
        nd = it[0].nodes
        trees.append([nd["feature_idx"].astype(int).tolist(), nd["num_threshold"].tolist(),
                      nd["left"].astype(int).tolist(), nd["right"].astype(int).tolist(),
                      nd["is_leaf"].astype(int).tolist(), nd["value"].tolist(),
                      nd["missing_go_to_left"].astype(int).tolist()])
    return {"base": float(np.ravel(mdl._baseline_prediction)[0]), "trees": trees}


def js_like(model_json, x):
    out = model_json["base"]
    for f, th, l, r, leaf, v, ml in model_json["trees"]:
        n = 0
        while not leaf[n]:
            xv = x[f[n]]
            n = (l[n] if ml[n] else r[n]) if np.isnan(xv) else (l[n] if xv <= th[n] else r[n])
        out += v[n]
    return out


models_json = {"median": export(med), "q10": export(q10), "q90": export(q90)}
diff = max(abs(js_like(models_json["median"], X[te][k]) - p[k]) for k in range(100))
print("tree-export max abs diff:", diff)
json.dump({"features": FEATURES, "lags": LAGS, "weather": WX, "pollutants": POLL, "metrics": res,
           "conformal_offsets": offsets, "buckets": BUCKETS, "trained_until": str(t.index[-1]),
           "backtest_24h": trace, "models": models_json}, open(os.path.join(OUT, "aqi_model.json"), "w"), separators=(",", ":"))
print("size MB", os.path.getsize(os.path.join(OUT, "aqi_model.json")) / 1e6)
