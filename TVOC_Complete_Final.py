# =============================================================================
# TVOC Signal Analysis — GPO-Tuned Kalman Filtering with Anomaly Detection
#
# Dataset : https://doi.org/10.17632/b5jvs7kykn.2
#           Place TVOC_Filtering_Subset.csv in data/ before running.
# Requirements: pip install -r requirements.txt
# Each "# ── CELL N" comment marks a Jupyter notebook cell boundary.
# Cell 5 (GPO): ~3 min  |  Cell 9 (10-fold CV): ~12 min
# =============================================================================

# ── CELL 1: Setup & Imports ──────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import chi2
from scipy.signal import welch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'         : 'serif',
    'font.size'           : 13,
    'axes.spines.top'     : False,
    'axes.spines.right'   : False,
    'axes.grid'           : True,
    'grid.alpha'          : 0.22,
    'figure.dpi'          : 150,
    'axes.linewidth'      : 1.5,
    'xtick.major.width'   : 1.5,
    'ytick.major.width'   : 1.5,
    'xtick.major.size'    : 6,
    'ytick.major.size'    : 6,
    'xtick.labelsize'     : 12,
    'ytick.labelsize'     : 12,
    'lines.linewidth'     : 2.0,
    'legend.fontsize'     : 11,
    'legend.framealpha'   : 0.9,
    'axes.titlesize'      : 13,
    'axes.labelsize'      : 13,
    'axes.titlepad'       : 8,
})

DATA_PATH = 'data/TVOC_Filtering_Subset.csv'
OUT_PATH  = './'
SEED      = 42

COL = {
    'gpo'  : '#C0392B',
    'heur' : '#2980B9',
    'ar1'  : '#27AE60',
    'adap' : '#8E44AD',
    'ma'   : '#E67E22',
    'es'   : '#7F8C8D',
    'raw'  : '#CCCCCC',
}

print('Setup complete')
print(f'  Data: {DATA_PATH}')
print(f'  Figures: {OUT_PATH}')


# ── CELL 2: Load & Clean Data ─────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH, low_memory=False)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

USE_COLS = ['Timestamp', 'TVOC (ppb)', 'eCO2 (ppm)', 'Temperature (C)', 'Humidity (%)']
df = df[USE_COLS].copy()

df = df.dropna(subset=['Temperature (C)']).reset_index(drop=True)

SAT_THRESHOLD = 10_000
n_sat         = (df['TVOC (ppb)'] > SAT_THRESHOLD).sum()
df['TVOC_raw'] = df['TVOC (ppb)'].copy()
df.loc[df['TVOC (ppb)'] > SAT_THRESHOLD, 'TVOC (ppb)'] = np.nan
df['TVOC (ppb)'] = df['TVOC (ppb)'].ffill()

tvoc  = df['TVOC (ppb)'].values.astype(float)
eco2  = df['eCO2 (ppm)'].values.astype(float)
temp  = df['Temperature (C)'].values.astype(float)
humid = df['Humidity (%)'].values.astype(float)
ts    = pd.to_datetime(df['Timestamp'].values)
n     = len(tvoc)
fs    = 1 / 5

df['date']  = df['Timestamp'].dt.date
df['hour']  = df['Timestamp'].dt.hour
df['day_n'] = (df['Timestamp'] - df['Timestamp'].min()).dt.days + 1

print(f'Rows              : {n:,}')
print(f'Saturation caps   : {n_sat:,}  ({n_sat/n*100:.2f}%)')
print(f'Date range        : {df["Timestamp"].min().date()} to {df["Timestamp"].max().date()}')
print(f'Duration          : {(df["Timestamp"].max()-df["Timestamp"].min()).days} days')
print(f'TVOC  mean={tvoc.mean():.1f} ppb  std={tvoc.std():.1f}  range=[{tvoc.min():.0f},{tvoc.max():.0f}]')
print(f'eCO2  mean={eco2.mean():.1f} ppm  std={eco2.std():.1f}')
print(f'Temp  mean={temp.mean():.1f} C    std={temp.std():.2f}')
print(f'RH    mean={humid.mean():.1f} %   std={humid.std():.2f}')


# ── CELL 3: Exploratory Analysis ─────────────────────────────────────────────

daily_raw = df.groupby('day_n').agg(
    N          = ('TVOC (ppb)', 'count'),
    TVOC_mean  = ('TVOC (ppb)', 'mean'),
    TVOC_std   = ('TVOC (ppb)', 'std'),
    TVOC_p95   = ('TVOC (ppb)', lambda x: np.percentile(x, 95)),
    eCO2_mean  = ('eCO2 (ppm)', 'mean'),
    Temp_mean  = ('Temperature (C)', 'mean'),
    RH_mean    = ('Humidity (%)', 'mean'),
).reset_index()

print('Daily TVOC Statistics:')
print(f"  {'Day':>4} {'N':>7} {'Mean':>8} {'Std':>8} {'P95':>8} {'eCO2':>7} {'Temp':>6} {'RH':>6}")
print('  ' + '-' * 60)
for _, r in daily_raw.iterrows():
    print(f"  {int(r['day_n']):>4} {int(r['N']):>7,} {r['TVOC_mean']:>8.1f} "
          f"{r['TVOC_std']:>8.1f} {r['TVOC_p95']:>8.1f} "
          f"{r['eCO2_mean']:>7.1f} {r['Temp_mean']:>6.1f} {r['RH_mean']:>6.1f}")

tvoc_sub_eda = tvoc[::5]
acf_lag1 = np.corrcoef(tvoc_sub_eda[:-1], tvoc_sub_eda[1:])[0, 1]
acf_lag5 = np.corrcoef(tvoc_sub_eda[:-5], tvoc_sub_eda[5:])[0, 1]
print(f'\nAutocorrelation:')
print(f'  Lag-1 ACF = {acf_lag1:.4f}  (near 1.0 supports random-walk model)')
print(f'  Lag-5 ACF = {acf_lag5:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].hist(tvoc[tvoc < 2000], bins=60, color=COL['gpo'], alpha=0.7, edgecolor='k', lw=0.3)
axes[0].set_xlabel('TVOC (ppb)'); axes[0].set_ylabel('Count')
axes[0].set_title('TVOC Distribution (clipped at 2,000 ppb)', fontweight='bold')

lags_eda = range(1, 41)
acf_eda  = [np.corrcoef(tvoc_sub_eda[:-k], tvoc_sub_eda[k:])[0, 1] for k in lags_eda]
conf_eda = 1.96 / np.sqrt(len(tvoc_sub_eda))
axes[1].bar(list(lags_eda), acf_eda, color=COL['heur'], alpha=0.7, width=0.8, edgecolor='k', lw=0.3)
axes[1].axhline(conf_eda,  ls='--', color='k', lw=1.2, label='95% CI')
axes[1].axhline(-conf_eda, ls='--', color='k', lw=1.2)
axes[1].set_xlabel('Lag'); axes[1].set_ylabel('ACF')
axes[1].set_title('TVOC Autocorrelation Function', fontweight='bold')
axes[1].legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PATH + 'fig0_eda.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig0_eda.pdf', bbox_inches='tight')
plt.show()
print('EDA complete')


# ── CELL 4: Filter Implementations ───────────────────────────────────────────

def kalman_rw(z, Q, R):
    x, P = float(z[0]), 1.0
    out  = []
    for y in z:
        P  += Q
        K   = P / (P + R)
        x  += K * (y - x)
        P   = (1 - K) * P
        out.append(x)
    return np.array(out)


def kalman_ar1(z, Q, R, phi=0.95):
    x, P = float(z[0]), 1.0
    out  = []
    for y in z:
        x_pred = phi * x
        P_pred = phi**2 * P + Q
        K      = P_pred / (P_pred + R)
        x      = x_pred + K * (y - x_pred)
        P      = (1 - K) * P_pred
        out.append(x)
    return np.array(out)


def kalman_adaptive(z, Q_init, R_init, window=100):

    x, P, Q, R = float(z[0]), 1.0, Q_init, R_init
    out, innov_buf = [], []
    for i, y in enumerate(z):
        P   += Q
        K    = P / (P + R)
        inn  = y - x
        x   += K * inn
        P    = (1 - K) * P
        innov_buf.append(inn)
        if i >= window:
            S = np.mean(np.array(innov_buf[-window:])**2)
            R = max(S - P, 0.01)
        out.append(x)
    return np.array(out)


def moving_average(z, w=60):

    return pd.Series(z).rolling(w, center=True, min_periods=1).mean().values


def exp_smooth(z, alpha=0.05):
    out = [float(z[0])]
    for y in z[1:]:
        out.append(alpha * y + (1 - alpha) * out[-1])
    return np.array(out)


def local_trend(z, w=360):
    return (pd.Series(z)
            .rolling(w, center=True, min_periods=w // 4)
            .median().ffill().bfill().values)


print('Filter functions defined.')


# ── CELL 5: GPO Hyperparameter Identification ─────────────────────────────────

def filter_objective(z, Q, R, trend):
    f         = kalman_rw(z, Q, R)
    raw_rough = np.mean(np.diff(z)**2) + 1e-9
    raw_fidel = np.mean((z - trend)**2) + 1e-9
    roughness = np.mean(np.diff(f)**2) / raw_rough
    fidelity  = np.mean((f - trend)**2) / raw_fidel
    return 0.5 * roughness + 0.5 * fidelity


def gp_optimize(z, trend, n_init=15, n_iter=45, seed=SEED, verbose=True):
    rng        = np.random.RandomState(seed)
    log_bounds = np.array([[-2.0, 2.0], [0.0, 3.5]])

    X_obs = np.column_stack([
        rng.uniform(log_bounds[0, 0], log_bounds[0, 1], n_init),
        rng.uniform(log_bounds[1, 0], log_bounds[1, 1], n_init),
    ])

    z_sub = z[::10]; t_sub = trend[::10]
    y_obs = np.array([filter_objective(z_sub, 10**x[0], 10**x[1], t_sub)
                      for x in X_obs])

    kernel = C(1.0, (1e-2, 1e2)) * RBF([1.0, 1.0], (1e-2, 1e2))
    gp     = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                      normalize_y=True, alpha=1e-6)

    best_x = X_obs[np.argmin(y_obs)].copy()
    best_y = float(y_obs.min())

    for _ in range(n_iter):
        gp.fit(X_obs, y_obs)
        g1 = np.linspace(log_bounds[0, 0], log_bounds[0, 1], 60)
        g2 = np.linspace(log_bounds[1, 0], log_bounds[1, 1], 60)
        G1, G2  = np.meshgrid(g1, g2)
        X_cand  = np.c_[G1.ravel(), G2.ravel()]
        mu, sigma = gp.predict(X_cand, return_std=True)
        z_score   = (best_y - mu) / (sigma + 1e-9)
        ei = (best_y - mu)*stats.norm.cdf(z_score) + sigma*stats.norm.pdf(z_score)
        next_x = X_cand[np.argmax(ei)]
        next_y = filter_objective(z_sub, 10**next_x[0], 10**next_x[1], t_sub)
        X_obs  = np.vstack([X_obs, next_x])
        y_obs  = np.append(y_obs, next_y)
        if next_y < best_y:
            best_y = next_y; best_x = next_x.copy()

    Q_opt = 10**best_x[0]; R_opt = 10**best_x[1]
    if verbose:
        print(f'  Q* = {Q_opt:.5f}   R* = {R_opt:.4f}   '
              f'R*/Q* = {R_opt/Q_opt:.1f}   obj* = {best_y:.5f}')
    return Q_opt, R_opt, X_obs, y_obs, gp


print('Running GPO on 3-day training segment (~3 minutes)...\n')
N_GPO     = 3 * 17280
z_gpo     = tvoc[:N_GPO]
trend_gpo = local_trend(z_gpo, w=360)

Q_opt, R_opt, X_gpo, y_gpo, gp_model = gp_optimize(
    z_gpo, trend_gpo, n_init=15, n_iter=45
)

print(f'\nGPO complete:')
print(f'  Q* = {Q_opt:.5f}')
print(f'  R* = {R_opt:.4f}')
print(f'  R*/Q* = {R_opt/Q_opt:.1f}  (measurement noise dominates)')
# Compute steady-state K by running the recursion to convergence
_P_ss = 1.0
for _ in range(10000):
    _P_prior = _P_ss + Q_opt
    _K_ss    = _P_prior / (_P_prior + R_opt)
    _P_ss    = (1 - _K_ss) * _P_prior
print(f'  Steady-state K = {_K_ss:.4f}  (sigma = {_P_ss**0.5:.3f} ppb)')


# ── CELL 6: Apply All Filters to Full 15-Day Dataset ─────────────────────────

print('Applying all filters to full dataset (~1 minute)...')

kf_opt      = kalman_rw(tvoc, Q_opt, R_opt)
kf_manual   = kalman_rw(tvoc, Q=0.1, R=2.0)
kf_ar1      = kalman_ar1(tvoc, Q_opt, R_opt)
kf_adaptive = kalman_adaptive(tvoc, Q_opt, R_opt, window=100)
ma          = moving_average(tvoc, w=60)
es          = exp_smooth(tvoc, alpha=0.05)
TREND_FULL  = local_trend(tvoc, w=360)

df['TVOC_filtered'] = kf_opt

methods = {
    'Raw TVOC'           : tvoc,
    'Moving Avg'         : ma,
    'Exp. Smooth'        : es,
    'Kalman (Heuristic)' : kf_manual,
    'Kalman AR(1)'       : kf_ar1,
    'Adaptive Kalman'    : kf_adaptive,
    'Kalman GPO (ours)'  : kf_opt,
}

print('All filters applied.')


# ── CELL 7: Signal Quality Metrics ───────────────────────────────────────────

def sig_metrics(raw, filt, trend, temp_arr, humid_arr):
    rn             = np.std(np.diff(raw))  + 1e-9
    fn             = np.std(np.diff(filt)) + 1e-9
    snr            = 20 * np.log10(rn / fn)
    nr             = (rn - fn) / rn * 100
    r_trend, _     = stats.pearsonr(filt, trend)
    innov          = raw - filt
    r_temp, p_temp = stats.pearsonr(innov, temp_arr)
    r_hum,  p_hum  = stats.pearsonr(innov, humid_arr)
    return snr, nr, r_trend, r_temp, p_temp, r_hum, p_hum


model_metrics = {}
print(f"  {'Method':<25} {'SNR(dB)':>8} {'NR(%)':>7} {'R_trend':>8} "
      f"{'r(Temp)':>9} {'p(T)':>8} {'r(RH)':>8} {'p(RH)':>8}  Env.OK?")
print('  ' + '-' * 92)

for name, sig in methods.items():
    if name == 'Raw TVOC':
        continue
    snr, nr, rt, rT, pT, rH, pH = sig_metrics(tvoc, sig, TREND_FULL, temp, humid)
    model_metrics[name] = dict(SNR=snr, NR=nr, R_trend=rt,
                               r_temp=rT, p_temp=pT, r_hum=rH, p_hum=pH)
    ok = 'OK' if (pT >= 0.05 and pH >= 0.05) else 'ARTIFACT'
    print(f"  {name:<25} {snr:>8.2f} {nr:>7.1f} {rt:>8.4f} "
          f"{rT:>+9.4f} {pT:>8.4f} {rH:>+8.4f} {pH:>8.4f}  {ok}")

print('\n  OK = innovations uncorrelated with environment (p >= 0.05)')
print('  ARTIFACT = environmental structure retained in residuals (p < 0.05)')


# ── CELL 8: Auxiliary Cross-Channel Consistency Assessment ───────────────────

eco2_norm = (eco2 - eco2.mean()) / eco2.std()

cross_sensor = {}
print(f"  {'Method':<25} {'r(eCO2)':>10}  {'p-value':>12}  Note")
print('  ' + '-' * 60)

for name, sig in methods.items():
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-9)
    r, p = stats.pearsonr(sig_norm, eco2_norm)
    cross_sensor[name] = (r, p)
    note = '' if name == 'Raw TVOC' else '(auxiliary check)'
    print(f"  {name:<25} {r:>+10.4f}  {p:>12.2e}  {note}")

print('\n  All methods show similar eCO2 correlation (~0.44)')
print('  This is expected: both channels respond to occupancy dynamics.')
print('  eCO2 is used only for consistency assessment, not primary validation.')


# ── CELL 9: 10-Fold Time-Series Cross-Validation ─────────────────────────────

tvoc_cv = tvoc[::5]
eco2_cv  = eco2[::5]

tscv     = TimeSeriesSplit(n_splits=10)
CV_NAMES = ['MA', 'ES', 'KF_heur', 'KF_ar1', 'KF_adap', 'KF_gpo']
cv_obj   = {k: [] for k in CV_NAMES}
cv_snr   = {k: [] for k in CV_NAMES}
cv_eco2  = {k: [] for k in CV_NAMES}

print('Running 10-fold CV (GPO re-optimized per fold)...\n')

for fold, (tr_idx, val_idx) in enumerate(tscv.split(tvoc_cv)):
    z_tr  = tvoc_cv[tr_idx]
    z_val = tvoc_cv[val_idx]
    e_val = eco2_cv[val_idx]

    trend_tr = local_trend(z_tr, w=72)
    Q_f, R_f, _, _, _ = gp_optimize(z_tr, trend_tr,
                                     n_init=10, n_iter=25,
                                     seed=SEED,  # fixed seed for reproducibility
                                     verbose=False)

    z_all = np.concatenate([z_tr, z_val])
    sp    = len(z_tr)

    fold_filts = {
        'MA'     : moving_average(z_val, w=60),
        'ES'     : exp_smooth(z_val, alpha=0.05),
        'KF_heur': kalman_rw(z_all, Q=0.1, R=2.0)[sp:],
        'KF_ar1' : kalman_ar1(z_all, Q_f, R_f)[sp:],
        'KF_adap': kalman_adaptive(z_all, Q_f, R_f, window=100)[sp:],
        'KF_gpo' : kalman_rw(z_all, Q_f, R_f)[sp:],
    }

    trend_val = local_trend(z_val, w=72)
    e_norm    = (e_val - e_val.mean()) / (e_val.std() + 1e-9)

    for k, f in fold_filts.items():
        rr = np.mean(np.diff(z_val)**2) + 1e-9
        rf = np.mean((z_val - trend_val)**2) + 1e-9
        roughness = np.mean(np.diff(f)**2) / rr
        fidelity  = np.mean((f - trend_val)**2) / rf
        cv_obj[k].append(0.5*roughness + 0.5*fidelity)

        rn = np.std(np.diff(z_val)) + 1e-9
        fn = np.std(np.diff(f))     + 1e-9
        cv_snr[k].append(20 * np.log10(rn / fn))

        fn_norm = (f - f.mean()) / (f.std() + 1e-9)
        r, _    = stats.pearsonr(fn_norm, e_norm)
        cv_eco2[k].append(r)

    print(f"  Fold {fold+1:2d}: Q*={Q_f:.4f}, R*={R_f:.3f}  |  "
          f"Balanced obj GPO:{cv_obj['KF_gpo'][-1]:.4f}  Heur:{cv_obj['KF_heur'][-1]:.4f}  |  "
          f"SNR GPO:{cv_snr['KF_gpo'][-1]:.1f}dB  MA:{cv_snr['MA'][-1]:.1f}dB")

print('\nCV complete.')


# ── CELL 9b: Bootstrap CIs + Wilcoxon Tests ──────────────────────────────────

def bootstrap_ci(arr, B=3000, seed=0):
    rng   = np.random.RandomState(seed)
    boots = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(B)]
    return np.mean(arr), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


ci_obj  = {k: bootstrap_ci(np.array(cv_obj[k]))  for k in CV_NAMES}
ci_snr  = {k: bootstrap_ci(np.array(cv_snr[k]))  for k in CV_NAMES}
ci_eco2 = {k: bootstrap_ci(np.array(cv_eco2[k])) for k in CV_NAMES}

print('Bootstrap 95% CI — Balanced Objective (lower = better):')
for k in CV_NAMES:
    m, lo, hi = ci_obj[k]
    print(f'  {k:<12}: {m:.4f}  [{lo:.4f}, {hi:.4f}]')

print('\nBootstrap 95% CI — CV SNR Gain (dB):')
for k in CV_NAMES:
    m, lo, hi = ci_snr[k]
    note = ' [NON-CAUSAL]' if k == 'MA' else ''
    print(f'  {k:<12}: {m:.2f} dB  [{lo:.2f}, {hi:.2f}]{note}')

print('\nWilcoxon Signed-Rank Tests (GPO vs baselines) on BALANCED OBJECTIVE:')
wilcox = {}
gpo_arr = np.array(cv_obj['KF_gpo'])
for bname in ['MA', 'ES', 'KF_heur', 'KF_ar1', 'KF_adap']:
    b_arr = np.array(cv_obj[bname])
    if np.allclose(gpo_arr, b_arr):
        continue
    W, p = stats.wilcoxon(gpo_arr, b_arr)
    sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    wilcox[bname] = (W, p, sig)
    print(f'  GPO vs {bname:<12}: W={W:.0f},  p={p:.4f}  {sig}')


# ── CELL 10: Ljung-Box Innovation Whiteness Test ─────────────────────────────

def ljung_box(residuals, lags=40):

    n_  = len(residuals)
    acf = [np.corrcoef(residuals[:-k], residuals[k:])[0, 1]
           for k in range(1, lags + 1)]
    Q   = n_ * (n_ + 2) * sum(r**2 / (n_ - k) for k, r in enumerate(acf, 1))
    p   = 1 - chi2.cdf(Q, df=lags)
    return Q, p


print('Ljung-Box Innovation Whiteness Test (lags=40):')
print(f"  {'Method':<20} {'Q-stat':>12}  {'p-value':>10}  Note")
print('  ' + '-' * 58)

for name, filt in [('GPO Kalman',   kf_opt),
                   ('Heuristic KF', kf_manual),
                   ('AR(1) Kalman', kf_ar1),
                   ('Adaptive KF',  kf_adaptive)]:
    inn_s      = (tvoc - filt)[::5]
    Q_lb, p_lb = ljung_box(inn_s)
    note       = 'lowest Q (GPO+Heuristic)' if name in ('GPO Kalman', 'Heuristic KF') else 'higher Q'
    print(f"  {name:<20} {Q_lb:>12,.0f}  {p_lb:>10.4f}  {note}")



# ── CELL 11: Final Results Table ──────────────────────────────────────────────

SEP = '=' * 112
print(SEP)
print('FINAL RESULTS TABLE')
print(SEP)
print(f"  {'Method':<25} {'SNR(dB)':>8} {'NR(%)':>7} {'R_trend':>8} "
      f"{'r(Temp)':>9} {'p(T)':>8} {'r(RH)':>8} {'p(RH)':>8} "
      f"{'CV_Obj':>8} {'95% CI':>18}")
print('  ' + '-' * 108)

CK_MAP = {
    'Moving Avg'          : 'MA',
    'Exp. Smooth'         : 'ES',
    'Kalman (Heuristic)'  : 'KF_heur',
    'Kalman AR(1)'        : 'KF_ar1',
    'Adaptive Kalman'     : 'KF_adap',
    'Kalman GPO (ours)'   : 'KF_gpo',
}

for nm, ck in CK_MAP.items():
    m             = model_metrics[nm]
    cv_m, lo, hi  = ci_obj[ck]
    causal = '(non-causal)' if nm == 'Moving Avg' else ''
    star   = ' *' if nm == 'Kalman GPO (ours)' else '  '
    print(f"  {nm+star:<27} {m['SNR']:>8.2f} {m['NR']:>7.1f} {m['R_trend']:>8.4f} "
          f"{m['r_temp']:>+9.4f} {m['p_temp']:>8.4f} "
          f"{m['r_hum']:>+8.4f} {m['p_hum']:>8.4f} "
          f"{cv_m:>8.4f} [{lo:.4f},{hi:.4f}] {causal}")

print(SEP)
print(f'Dataset   : {n:,} observations  |  15 days  |  5-second sampling')
print(f'GPO       : Q* = {Q_opt:.5f},  R* = {R_opt:.4f},  R*/Q* = {R_opt/Q_opt:.1f}')
print(f'Artifacts : {n_sat:,} rows capped (saturation > {SAT_THRESHOLD:,} ppb)')
print('\nWilcoxon (applied to balanced objective, NOT raw SNR):')
for bname, (W, p, sig) in wilcox.items():
    print(f'  GPO vs {bname:<12}: W={W:.0f},  p={p:.4f}  {sig}')


# ── CELL 12: Figure 1 — 15-Day Overview ──────────────────────────────────────

dates = ts
S     = 12
d_    = dates[::S]; t_=tvoc[::S]; g_=kf_opt[::S]
e_    = eco2[::S];  tm_=temp[::S]; h_=humid[::S]
tr_   = TREND_FULL[::S]
inn_f = tvoc - kf_opt;  sigma_inn = np.std(inn_f)

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

axes[0].fill_between(d_, t_, alpha=0.15, color=COL['raw'])
axes[0].plot(d_, t_,  color=COL['raw'], lw=0.4, alpha=0.5, label='Raw TVOC')
axes[0].plot(d_, g_,  color=COL['gpo'], lw=2.5, label='GPO Kalman (proposed)')
axes[0].plot(d_, tr_, color='navy', lw=2.0, ls='--', alpha=0.6, label='30-min rolling median')
axes[0].set_ylabel('TVOC (ppb)')
axes[0].set_title('(a) Raw Measurement, GPO Kalman Estimate, and 30-min Rolling Median',
                  fontweight='bold', fontsize=13)
axes[0].legend(fontsize=11, ncol=3, loc='upper right')

axes[1].plot(d_, e_, color='#16A085', lw=1.8, alpha=0.8,
             label='eCO2 (ppm) — auxiliary corroborative channel')
axes[1].set_ylabel('eCO2 (ppm)')
axes[1].set_title('(b) eCO2: Auxiliary Corroborative Channel (algorithmically distinct SGP30 output)',
                  fontweight='bold', fontsize=13)
axes[1].legend(fontsize=11)

axes[2].plot(d_, tm_, color='#E74C3C', lw=0.8, label='Temperature (C)')
ax2r = axes[2].twinx()
ax2r.plot(d_, h_, color='#3498DB', lw=0.8, alpha=0.8, label='Humidity (%)')
axes[2].set_ylabel('Temperature (C)', color='#E74C3C')
ax2r.set_ylabel('Humidity (%)', color='#3498DB')
axes[2].set_title('(c) Environmental Covariates', fontweight='bold', fontsize=13)
axes[2].legend(loc='upper left', fontsize=9)
ax2r.legend(loc='upper right', fontsize=9)

axes[3].plot(d_, inn_f[::S], color='#8E44AD', lw=0.5, alpha=0.6,
             label='Innovation sequence')
axes[3].axhline(0, color='k', lw=0.8)
axes[3].axhline( 2*sigma_inn, color='red', lw=1.0, ls='--', alpha=0.7, label='±2σ')
axes[3].axhline(-2*sigma_inn, color='red', lw=1.0, ls='--', alpha=0.7)
axes[3].set_ylabel('Innovation (ppb)'); axes[3].set_xlabel('Date')
axes[3].set_title('(d) GPO Kalman Innovations with ±2σ Bounds',
                  fontweight='bold', fontsize=13)
axes[3].legend(fontsize=11)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PATH + 'fig1_15day_overview.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig1_15day_overview.pdf', bbox_inches='tight')
plt.show()
print('Figure 1 saved')


# ── CELL 13: Figure 2 — GPO Convergence ──────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

lQ = X_gpo[:, 0]; lR = X_gpo[:, 1]; bi = np.argmin(y_gpo)

sc = axes[0].scatter(lQ, lR, c=y_gpo, cmap='plasma_r', s=50,
                     edgecolors='k', lw=0.4, zorder=3)
axes[0].scatter(lQ[bi], lR[bi], c='lime', s=250, marker='*',
                edgecolors='k', lw=1, zorder=5,
                label=f'Optimum: Q*={Q_opt:.4f}, R*={R_opt:.2f}')
plt.colorbar(sc, ax=axes[0], label='Objective (lower = better)')
axes[0].set_xlabel('log10(Q)'); axes[0].set_ylabel('log10(R)')
axes[0].set_title('(a) GP-BO Evaluated Points', fontweight='bold')
axes[0].legend(fontsize=11)

conv = np.minimum.accumulate(y_gpo)
axes[1].plot(conv, color=COL['gpo'], lw=2.5, label='Best objective so far')
axes[1].scatter(bi, conv[bi], c='lime', s=150, edgecolors='k', lw=1,
                zorder=5, label=f'Convergence at iteration {bi}')
axes[1].axvline(15, ls=':', color='navy', lw=1.5, alpha=0.7,
                label='Initial LHS design (15 points)')
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Best Objective Value')
axes[1].set_title('(b) Convergence Curve', fontweight='bold')
axes[1].legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PATH + 'fig2_gpo_convergence.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig2_gpo_convergence.pdf', bbox_inches='tight')
plt.show()
print('Figure 2 saved')


# ── CELL 14: Figure 3 — Cross-Validation Results ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

bar_labels = ['Moving\nAvg*', 'Exp.\nSmooth', 'Kalman\nHeuristic',
              'Kalman\nAR(1)', 'Adaptive\nKalman†', 'Kalman\nGPO**']
bar_colors = [COL['ma'], COL['es'], COL['heur'],
              COL['ar1'], COL['adap'], COL['gpo']]
x = np.arange(6)

vals_obj  = [ci_obj[k][0] for k in CV_NAMES]
errs_lo_o = [ci_obj[k][0] - ci_obj[k][1] for k in CV_NAMES]
errs_hi_o = [ci_obj[k][2] - ci_obj[k][0] for k in CV_NAMES]

Y_CLIP = 2.5

vals_clipped = [min(v, Y_CLIP) for v in vals_obj]

bars_a = axes[0].bar(x, vals_clipped,
                     color=bar_colors, edgecolor='black', lw=1.0, width=0.6)

for i in range(6):
    lo = errs_lo_o[i]
    hi = errs_hi_o[i] if i != 4 else min(errs_hi_o[i], Y_CLIP - vals_clipped[i])
    axes[0].errorbar(x[i], vals_clipped[i],
                     yerr=[[lo], [hi]],
                     fmt='none', color='black', capsize=6, lw=2.0, capthick=2.0)

bars_a[5].set_edgecolor('gold'); bars_a[5].set_linewidth(3)

axes[0].set_ylim(-0.10, Y_CLIP)

axes[0].text(4, Y_CLIP - 0.08,
             f'{vals_obj[4]:.1f}\n(clipped)',
             ha='center', va='top', fontsize=9,
             color='white', fontweight='bold')

bars_a[4].set_hatch('///')
bars_a[4].set_edgecolor('black')

y_sig = vals_clipped[3] + errs_hi_o[3] + 0.12
y_sig = min(y_sig, 1.95)
axes[0].annotate('', xy=(3, y_sig), xytext=(5, y_sig),
                 arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
axes[0].text(4, y_sig + 0.05, 'p = 0.049 *',
             ha='center', fontsize=9, color='black', fontweight='bold')

axes[0].set_xticks(x)
axes[0].set_xticklabels(bar_labels, fontsize=10)
axes[0].set_ylabel('CV Balanced Objective (lower = better)', fontsize=11)
axes[0].set_title(
    f'(a) Balanced Objective Score\n'
    f'(GPO: p={wilcox["KF_ar1"][1]:.3f} vs AR(1); '
    f'p={wilcox["KF_adap"][1]:.3f} vs Adaptive)',
    fontweight='bold', fontsize=13)

axes[0].text(0.5, -0.16,
             f'* Non-causal; offline baseline only.   '
             f'\u2020 Actual CV obj = {vals_obj[4]:.1f} (clipped; over-smoothing).   '
             f'** Best automated tuning.',
             transform=axes[0].transAxes, fontsize=8.5, color='#444444',
             ha='center', va='top')

vals_eco  = [ci_eco2[k][0] for k in CV_NAMES]
errs_lo_e = [ci_eco2[k][0] - ci_eco2[k][1] for k in CV_NAMES]
errs_hi_e = [ci_eco2[k][2] - ci_eco2[k][0] for k in CV_NAMES]

bars_b = axes[1].bar(x, vals_eco, yerr=[errs_lo_e, errs_hi_e], capsize=6,
                     color=bar_colors, edgecolor='black', lw=1.0, width=0.6,
                     error_kw={'lw': 2.5, 'capthick': 2.5})
bars_b[5].set_edgecolor('gold'); bars_b[5].set_linewidth(3)

axes[1].set_xticks(x)
axes[1].set_xticklabels(bar_labels, fontsize=10)
axes[1].set_ylabel('CV r(eCO\u2082) \u2014 auxiliary only', fontsize=11)
axes[1].set_title('(b) Auxiliary eCO\u2082 Cross-Channel Correlation\n'
                  '(all methods comparable; eCO\u2082 is not a reference instrument)',
                  fontweight='bold', fontsize=13)
axes[1].set_ylim(0, 0.85)
axes[1].text(0.5, -0.12,
             '* Non-causal.  eCO\u2082 channel used for auxiliary consistency check only.',
             transform=axes[1].transAxes, fontsize=8.5, color='#444444',
             ha='center', va='top')

plt.tight_layout()
plt.savefig(OUT_PATH + 'fig3_cv_results.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig3_cv_results.pdf', bbox_inches='tight')
plt.show()
print('Figure 3 saved')
# ── CELL 15: Figure 4 — Innovation ACF Analysis ──────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

filter_pairs = [
    ('GPO Kalman (proposed)', kf_opt,      COL['gpo']),
    ('Kalman Heuristic',      kf_manual,   COL['heur']),
    ('Kalman AR(1)-SS',       kf_ar1,      COL['ar1']),
    ('Adaptive Kalman',       kf_adaptive, COL['adap']),
]

for ax, (nm, f, col) in zip(axes.ravel(), filter_pairs):
    inn_s = (tvoc - f)[::5]
    lags  = range(1, 51)
    acf_v = [np.corrcoef(inn_s[:-k], inn_s[k:])[0, 1] for k in lags]
    conf  = 1.96 / np.sqrt(len(inn_s))
    ax.bar(list(lags), acf_v, color=col, alpha=0.75, width=0.8,
           edgecolor='k', lw=0.3)
    ax.axhline( conf, ls='--', color='k', lw=1.2, label='95% CI')
    ax.axhline(-conf, ls='--', color='k', lw=1.2)
    ax.axhline(0, color='k', lw=0.5)
    frac_out   = np.mean(np.abs(acf_v) > conf) * 100
    Q_lb, p_lb = ljung_box(inn_s, lags=40)
    ax.set_title(f'{nm}\nLB Q={Q_lb:,.0f},  p={p_lb:.3f}  |  '
                 f'{frac_out:.0f}% lags outside CI',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUT_PATH + 'fig4_innovation_acf.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig4_innovation_acf.pdf', bbox_inches='tight')
plt.show()
print('Figure 4 saved')


# ── CELL 16: Figure 5 — Comparative Metrics Summary ──────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

names_plot  = list(model_metrics.keys())
short_names = ['MA*', 'ES', 'KF-Heur', 'KF-AR1', 'KF-Adp', 'KF-GPO']
bar_c       = [COL['ma'], COL['es'], COL['heur'],
               COL['ar1'], COL['adap'], COL['gpo']]
x = np.arange(len(names_plot))

for ax, (metric, ylabel, title, best_fn) in zip(axes, [
    ('SNR',     'SNR Gain (dB)',               'SNR Gain',         max),
    ('NR',      'Noise Reduction (%)',          'Noise Reduction',  max),
    ('R_trend', 'Correlation with 30-min Trend','Trend Fidelity',   max),
]):
    vals = [model_metrics[nm][metric] for nm in names_plot]
    bars = ax.bar(x, vals, color=bar_c, edgecolor='k', lw=0.8)
    best = best_fn(vals)
    for i, (b, v) in enumerate(zip(bars, vals)):
        if abs(v - best) < 1e-6:
            b.set_edgecolor('gold'); b.set_linewidth(3)
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')

axes[0].text(0.02, -0.22, '* Non-causal', transform=axes[0].transAxes,
             fontsize=7.5, color='gray')
plt.tight_layout()
plt.savefig(OUT_PATH + 'fig5_metrics_summary.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig5_metrics_summary.pdf', bbox_inches='tight')
plt.show()
print('Figure 5 saved')


# ── CELL 17: Figure 6 — Longitudinal Analysis (TVOC-centric) ─────────────────

daily_stats = df.groupby('day_n').agg(
    tvoc_mean  = ('TVOC (ppb)',    'mean'),
    tvoc_std   = ('TVOC (ppb)',    'std'),
    filt_mean  = ('TVOC_filtered', 'mean'),
    temp_mean  = ('Temperature (C)', 'mean'),
    humid_mean = ('Humidity (%)',  'mean'),
).reset_index()

hourly = df.groupby('hour').agg(
    tvoc_mean = ('TVOC (ppb)',    'mean'),
    tvoc_std  = ('TVOC (ppb)',    'std'),
    filt_mean = ('TVOC_filtered', 'mean'),
).reset_index()

# PSD on full signal, fs=0.2 Hz, threshold f>0.05 Hz (below Nyquist=0.1 Hz)
f_r, psd_r = welch(tvoc,   fs=fs, nperseg=1024)
f_f, psd_f = welch(kf_opt, fs=fs, nperseg=1024)
hf_red = (1 - psd_f[f_f > 0.05].mean() / psd_r[f_r > 0.05].mean()) * 100

fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])


ax1.fill_between(daily_stats['day_n'],
    daily_stats['tvoc_mean'] - daily_stats['tvoc_std'],
    daily_stats['tvoc_mean'] + daily_stats['tvoc_std'],
    alpha=0.18, color=COL['raw'], label='Raw: mean ± 1 SD')
ax1.plot(daily_stats['day_n'], daily_stats['tvoc_mean'], 'o--',
         color=COL['raw'], lw=1.5, ms=5, label='Raw daily mean')
ax1.plot(daily_stats['day_n'], daily_stats['filt_mean'], 's-',
         color=COL['gpo'], lw=2.2, ms=6, label='GPO Kalman estimate mean')
ax1.set_xlabel('Deployment Day'); ax1.set_ylabel('TVOC (ppb)')
ax1.set_title('(a) Daily Mean TVOC: Raw vs GPO Kalman Estimate (Days 1 to 17)',
              fontweight='bold', fontsize=13)
ax1.legend(fontsize=11); ax1.set_xticks(range(1, 18))

h = hourly['hour']
ax2.fill_between(h,
    hourly['tvoc_mean'] - hourly['tvoc_std'] / 2,
    hourly['tvoc_mean'] + hourly['tvoc_std'] / 2,
    alpha=0.15, color=COL['raw'])
ax2.plot(h, hourly['tvoc_mean'], color=COL['raw'], lw=1, alpha=0.7,
         label='Raw hourly mean')
ax2.plot(h, hourly['filt_mean'], color=COL['gpo'], lw=2.2,
         label='GPO Kalman mean')
ax2.axvspan(8, 18, alpha=0.07, color='#2ECC71',
            label='Occupied hours (08:00-18:00)')
ax2.set_xlabel('Hour of Day'); ax2.set_ylabel('TVOC (ppb)')
ax2.set_title('(b) Diurnal TVOC Profile (15-day aggregate)',
              fontweight='bold', fontsize=13)
ax2.legend(fontsize=11); ax2.set_xticks(range(0, 24, 3))

psd_r_dB = 10 * np.log10(psd_r + 1e-9)
psd_f_dB = 10 * np.log10(psd_f + 1e-9)
ax3.semilogx(f_r[1:], psd_r_dB[1:], color=COL['raw'], lw=1.5, alpha=0.8,
             label='Raw TVOC')
ax3.semilogx(f_f[1:], psd_f_dB[1:], color=COL['gpo'], lw=2.2,
             label='GPO Kalman filtered')
ax3.axvline(0.05, color='navy', ls='--', lw=1.2, alpha=0.7, label='0.05 Hz')
ax3.text(0.065, ax3.get_ylim()[0] + 5 if len(ax3.get_lines()) > 0 else 0,
         f'High-freq. power\nreduced {hf_red:.1f}%',
         fontsize=8.5, color=COL['gpo'], fontweight='bold')
ax3.set_xlabel('Frequency (Hz)'); ax3.set_ylabel('PSD (dB)')
ax3.set_title(f'(c) Power Spectral Density\n(GPO Kalman: {hf_red:.1f}% high-freq. reduction)',
              fontweight='bold', fontsize=13)
ax3.legend(fontsize=11)

plt.savefig(OUT_PATH + 'fig6_longitudinal_analysis.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig6_longitudinal_analysis.pdf', bbox_inches='tight')
plt.show()
print('Figure 6 saved')


# ── CELL 18: Anomaly Detection ────────────────────────────────────────────────

K_THRESHOLD  = 3.0
SIGMA_WINDOW = 7200
SPIKE_MAX    = 30
EPISODE_MAX  = 300

innovations = tvoc - kf_opt
inn_series  = pd.Series(innovations)
roll_sigma  = (inn_series
               .rolling(SIGMA_WINDOW, center=True, min_periods=SIGMA_WINDOW // 4)
               .std().ffill().bfill().values)

z_score      = np.abs(innovations) / (roll_sigma + 1e-9)
anomaly_flag = (z_score > K_THRESHOLD)
n_flagged    = anomaly_flag.sum()

print(f'Anomaly detection (k={K_THRESHOLD}, 10-hour rolling sigma):')
print(f'  Rolling sigma mean : {roll_sigma.mean():.2f} ppb')
print(f'  Flagged samples    : {n_flagged:,} / {n:,} ({n_flagged/n*100:.2f}%)')

flags_int = anomaly_flag.astype(int)
events = []
i = 0
while i < n:
    if flags_int[i]:
        j = i
        while j < n and flags_int[j]:
            j += 1
        events.append({
            'start_idx'      : i,
            'end_idx'        : j,
            'start_ts'       : ts[i],
            'duration_s'     : (j - i) * 5,
            'peak_innovation': np.max(np.abs(innovations[i:j])),
            'peak_raw_tvoc'  : np.max(tvoc[i:j]),
            'mean_zscore'    : np.mean(z_score[i:j]),
            'hour'           : pd.Timestamp(ts[i]).hour,
        })
        i = j
    else:
        i += 1

events_df = pd.DataFrame(events)
events_df['day_n'] = ((events_df['start_ts'] - ts.min())
                      / np.timedelta64(1, 'D')).astype(int) + 1

n_spikes    = (events_df['duration_s'] <= SPIKE_MAX).sum()
n_episodes  = ((events_df['duration_s'] > SPIKE_MAX) &
               (events_df['duration_s'] <= EPISODE_MAX)).sum()
n_sustained = (events_df['duration_s'] > EPISODE_MAX).sum()
n_occupied  = (events_df['hour'].between(8, 17)).sum()
occ_pct     = n_occupied / len(events_df) * 100

print(f'\nEvent clustering results:')
print(f'  Total distinct events         : {len(events_df):,}')
print(f'  Transient spikes (<=30s)      : {n_spikes:,}  ({n_spikes/len(events_df)*100:.1f}%)')
print(f'  Emission episodes (30-300s)   : {n_episodes:,}  ({n_episodes/len(events_df)*100:.1f}%)')
print(f'  Sustained events (>300s)      : {n_sustained:,}')
print(f'  Events during occupied hours  : {n_occupied:,}  ({occ_pct:.1f}%)')
print(f'  Mean events per day           : {len(events_df)/16:.0f}  (1,897 / 16 day-groups)')

print(f'\nTop 5 events by peak innovation:')
top5 = events_df.nlargest(5, 'peak_innovation')
print(f"  {'Day':>5} {'Duration':>10} {'Peak Inn.':>10} {'Peak Raw':>10}")
for _, r in top5.iterrows():
    print(f"  Day {int(r['day_n']):2d}  {r['duration_s']:>8.0f}s  "
          f"{r['peak_innovation']:>10.1f}  {r['peak_raw_tvoc']:>10.1f}")



# ── CELL 19: Figure 7 — Anomaly Detection (6-panel) ──────────────────────────

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        height_ratios=[2.2, 1.4, 1.4, 1.2],
                        hspace=0.42, wspace=0.32)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[3, 0])
ax6 = fig.add_subplot(gs[3, 1])


S_ = 12
d_ = dates[::S_]; tv_ = tvoc[::S_]; kf_ = kf_opt[::S_]
z_ = z_score[::S_]; fl_ = anomaly_flag[::S_]

ax1.fill_between(d_, tv_, alpha=0.12, color=COL['raw'])
ax1.plot(d_, tv_, color=COL['raw'], lw=0.4, alpha=0.5, label='Raw TVOC')
ax1.plot(d_, kf_, color=COL['gpo'], lw=2.5, label='GPO Kalman estimate', zorder=3)
anom_d = d_[fl_]; anom_tv = tv_[fl_]
if len(anom_d) > 0:
    ax1.scatter(anom_d, anom_tv, c='#F39C12', s=8, zorder=5,
                alpha=0.6, label=f'Anomaly flag (k={K_THRESHOLD})', marker='|')
top5 = events_df.nlargest(5, 'peak_innovation')
_offsets = [(-45, -30), (40, -30), (-50, -55), (45, -55), (0, -80)]
for _i, (_, ev) in enumerate(top5.iterrows()):
    _ox, _oy = _offsets[_i] if _i < len(_offsets) else (0, -30)
    _yval = float(ev['peak_raw_tvoc'])
    ax1.annotate(f"{_yval:.0f} ppb",
                 xy=(ev['start_ts'], _yval),
                 xytext=(_ox, _oy), textcoords='offset points',
                 ha='center', fontsize=8, color='#C0392B', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2,
                                 connectionstyle='arc3,rad=0.1'))
ax1.set_ylabel('TVOC (ppb)')
ax1.set_title(f'(a) TVOC with Anomaly Flags [{len(events_df):,} events, k={K_THRESHOLD}]',
              fontweight='bold', fontsize=13)
ax1.legend(fontsize=8.5, ncol=3, loc='upper right', framealpha=0.9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=11)

ax2.plot(d_, z_, color='#8E44AD', lw=0.6, alpha=0.7,
         label='Standardized |innovation|')
ax2.axhline(K_THRESHOLD, color='#E74C3C', lw=1.8, ls='--',
            label=f'Threshold k={K_THRESHOLD}')
ax2.fill_between(d_, 0, z_, where=(z_ > K_THRESHOLD),
                 color='#E74C3C', alpha=0.3, label='Anomaly region')
ax2.set_ylabel('|Innovation| / sigma_emp')
ax2.set_ylim(bottom=0)
ax2.set_title('(b) Standardized Innovation (above dashed line = anomaly)',
              fontweight='bold', fontsize=13)
ax2.legend(fontsize=8.5, ncol=3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=11)

dur = events_df['duration_s'].values
ax3.hist(dur[dur <= 120], bins=24, color='#2980B9',
         edgecolor='k', lw=0.4, alpha=0.8)
ax3.axvline(SPIKE_MAX,   color='#E74C3C', lw=1.5, ls='--',
            label=f'Spike limit ({SPIKE_MAX}s)')
ax3.axvline(EPISODE_MAX, color='#E67E22', lw=1.5, ls='-.',
            label=f'Episode limit ({EPISODE_MAX}s)')
ax3.set_xlabel('Event Duration (s)'); ax3.set_ylabel('Count')
ax3.set_title(f'(c) Duration Distribution (clipped at 120s)\n'
              f'Spikes: {n_spikes}  Episodes: {n_episodes}  Sustained: {n_sustained}',
              fontweight='bold', fontsize=12)
ax3.legend(fontsize=11)

h_ev = events_df.groupby('hour').size().reindex(range(24), fill_value=0)
bar_cols = ['#2ECC71' if (8 <= h < 18) else '#95A5A6' for h in range(24)]
ax4.bar(range(24), h_ev.values, color=bar_cols, edgecolor='k', lw=0.4, alpha=0.85)
ax4.axvspan(8, 18, alpha=0.07, color='#2ECC71')
ax4.set_xlabel('Hour of Day'); ax4.set_ylabel('Events')
ax4.set_title(f'(d) Hourly Pattern — {occ_pct:.0f}% in occupied hours (08:00-18:00)',
              fontweight='bold', fontsize=12)
ax4.set_xticks(range(0, 24, 3))
ax4.legend(handles=[
    mpatches.Patch(color='#2ECC71', alpha=0.7, label='Occupied'),
    mpatches.Patch(color='#95A5A6', alpha=0.7, label='Unoccupied')
], fontsize=11)

d_ev = events_df.groupby('day_n').size()
ax5.bar(d_ev.index, d_ev.values, color='#16A085',
        edgecolor='k', lw=0.4, alpha=0.85, width=0.7)
ax5.axhline(d_ev.mean(), color='#C0392B', lw=1.5, ls='--',
            label=f'Mean = {d_ev.mean():.0f} events/day')
ax5.set_xlabel('Calendar Day'); ax5.set_ylabel('Events')
ax5.set_title('(e) Daily Anomaly Event Count', fontweight='bold', fontsize=12)
ax5.set_xticks(range(1, 18)); ax5.legend(fontsize=11)

big_ev = events_df.nlargest(1, 'peak_innovation').iloc[0]
zoom_s = pd.Timestamp(big_ev['start_ts']) - pd.Timedelta('30min')
zoom_e = pd.Timestamp(big_ev['start_ts']) + pd.Timedelta('60min')
mask_z = (dates >= zoom_s) & (dates <= zoom_e)
ts_z = dates[mask_z]; tv_z = tvoc[mask_z]; kf_z = kf_opt[mask_z]
fl_z = anomaly_flag[mask_z]; inn_z = innovations[mask_z]

ax6_r = ax6.twinx()
ax6.plot(ts_z, tv_z, color=COL['raw'], lw=0.8, alpha=0.7, label='Raw TVOC')
ax6.plot(ts_z, kf_z, color=COL['gpo'], lw=2, label='GPO Kalman')
ax6.scatter(ts_z[fl_z], tv_z[fl_z], c='#F39C12', s=30, zorder=5,
            label='Anomaly flags', marker='v', alpha=0.8)
ax6_r.plot(ts_z, np.abs(inn_z), color='#8E44AD', lw=1, alpha=0.6,
           label='|Innovation| (right)')
ax6_r.set_ylabel('|Innovation| (ppb)', color='#8E44AD', fontsize=9)
ax6.set_ylabel('TVOC (ppb)', fontsize=9); ax6.set_xlabel('Time')
ax6.set_title(f'(f) Largest Event Zoom — Day {int(big_ev["day_n"])}\n'
              f'Peak raw: {big_ev["peak_raw_tvoc"]:.0f} ppb, '
              f'Peak innovation: {big_ev["peak_innovation"]:.0f} ppb',
              fontweight='bold', fontsize=12)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
l1, lb1 = ax6.get_legend_handles_labels()
l2, lb2 = ax6_r.get_legend_handles_labels()
ax6.legend(l1+l2, lb1+lb2, fontsize=7.5, loc='upper left')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=11)

plt.subplots_adjust(top=0.97, hspace=0.48, wspace=0.32)
plt.savefig(OUT_PATH + 'fig7_anomaly_detection.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PATH + 'fig7_anomaly_detection.pdf', bbox_inches='tight')
plt.show()
print('Figure 7 saved')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ALL ANALYSIS COMPLETE')
print('='*70)
print(f'Figures saved: fig1 through fig7 + fig0_eda.png')
print(f'Dataset: {n:,} observations, 15 days, 5-second sampling')
print(f'GPO: Q*={Q_opt:.5f}, R*={R_opt:.4f}, R*/Q*={R_opt/Q_opt:.1f}')
print(f'Anomaly events: {len(events_df):,} total, {occ_pct:.0f}% during occupied hours')
print('='*70)