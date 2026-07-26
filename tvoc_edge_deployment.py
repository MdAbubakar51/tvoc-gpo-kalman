# =============================================================================
# Real-Time TVOC Edge Deployment — Raspberry Pi 5
# GPO-Tuned Kalman Filter with Live Dashboard
# Hardware:
#   Raspberry Pi 5 (4 GB or 8 GB)
#   Sensirion SGP30  — I2C — primary TVOC / eCO2 sensor
#   Bosch BME688     — I2C — temperature and humidity
#
# Requirements:
#   pip install adafruit-circuitpython-sgp30 adafruit-blinka
#   pip install adafruit-circuitpython-bme680
#   pip install dash plotly pandas
#
# Usage:
#   python tvoc_edge_deployment.py
#   Dashboard: http://<raspberry-pi-ip>:8050
# =============================================================================

import time
import os
import json
import csv
import board
import busio
import adafruit_sgp30
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from collections import deque


# =============================================================================
# SECTION 1 — PARAMETERS
# =============================================================================

# GPO-identified noise covariance parameters (Q*, R*)
# Identified offline from 269,010 observations using the balanced objective.
Q_OPT         = 6.021     # process noise covariance
R_OPT         = 60.21     # measurement noise covariance
SAT_THRESHOLD = 10_000    # ppb — SGP30 firmware ceiling

# Anomaly detection
ANOMALY_K      = 3.0      # standardized innovation threshold
ANOMALY_WINDOW = 7200     # 10-hour rolling window (samples)


# =============================================================================
# SECTION 2 — KALMAN FILTER STATE
# =============================================================================

x_est     = 0.0
P_cov     = 1.0
prev_tvoc = None
n_sat     = 0

innovation_history = deque(maxlen=ANOMALY_WINDOW)
n_anomaly          = 0


def kalman_update(z_raw):
    """
    Single-sample update of the GPO-tuned random-walk Kalman filter.

    State-space model:
        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
        y_t = x_t     + v_t,   v_t ~ N(0, R)

    Returns: x_filt, K_gain, P_post, z_used, sat_flag, anomaly, latency_ms
    """
    global x_est, P_cov, prev_tvoc, n_sat, n_anomaly

    t0 = time.perf_counter()

    # Saturation artifact — forward-fill from last valid reading
    sat_flag = False
    if z_raw > SAT_THRESHOLD:
        sat_flag  = True
        n_sat    += 1
        z_used    = prev_tvoc if prev_tvoc is not None else 0.0
        print(f"[ARTIFACT] {z_raw} ppb saturated → forward-filled {z_used:.1f} ppb")
    else:
        z_used = float(z_raw)

    prev_tvoc = z_used

    # Kalman recursion
    P_prior = P_cov + Q_OPT
    K_gain  = P_prior / (P_prior + R_OPT)
    innov   = z_used - x_est
    x_est   = x_est + K_gain * innov
    P_cov   = (1 - K_gain) * P_prior

    # Anomaly detection
    innovation_history.append(abs(innov))
    if len(innovation_history) >= 100:
        sigma_emp = pd.Series(list(innovation_history)).std()
        anomaly   = (abs(innov) / (sigma_emp + 1e-9)) > ANOMALY_K
        if anomaly:
            n_anomaly += 1
    else:
        anomaly = False

    latency_ms = (time.perf_counter() - t0) * 1000

    return x_est, K_gain, P_cov, z_used, sat_flag, anomaly, latency_ms


# =============================================================================
# SECTION 3 — SENSOR SETUP
# =============================================================================

print("Initializing sensors via I2C bus...")
i2c   = busio.I2C(board.SCL, board.SDA)

sgp30 = adafruit_sgp30.Adafruit_SGP30(i2c)
sgp30.iaq_init()

# BME688 — uncomment after installing adafruit-circuitpython-bme680
# import adafruit_bme680
# bme688 = adafruit_bme680.Adafruit_BME680_I2C(i2c)

BASELINE_FILE = "sgp30_baseline.json"


def save_baseline():
    """Save SGP30 learned baseline registers to disk."""
    baseline = {
        "eCO2_baseline": sgp30.baseline_eCO2,
        "TVOC_baseline": sgp30.baseline_TVOC,
        "saved_at":      datetime.now().isoformat()
    }
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)


def load_baseline():
    """Restore SGP30 baseline from disk if available."""
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, 'r') as f:
            b = json.load(f)
        sgp30.set_iaq_baseline(b["eCO2_baseline"], b["TVOC_baseline"])
        print(f"SGP30 baseline restored from {b['saved_at']}")
    else:
        print("No saved baseline — SGP30 will self-calibrate (~12 h)")


load_baseline()

print("SGP30 warming up (15 s)...")
time.sleep(15)
print("Sensor ready.\n")


# =============================================================================
# SECTION 4 — LOGGING
# =============================================================================

LOG_CSV             = "tvoc_realtime_log.csv"
LOG_EXCEL           = "tvoc_realtime_log.xlsx"
EXCEL_BACKUP_EVERY  = 120   # samples (~10 min)
BASELINE_SAVE_EVERY = 12    # samples (~60 s)

LOG_COLS = [
    'Timestamp', 'TVOC_raw', 'TVOC_filtered', 'eCO2_raw',
    'Temperature_C', 'Humidity_pct', 'Kalman_gain', 'P_error',
    'Innovation', 'Sat_flag', 'Anomaly_flag', 'Latency_ms',
]

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(LOG_COLS)
    print(f"Log created: {LOG_CSV}")
else:
    print(f"Appending to: {LOG_CSV}")

BUFFER_SIZE = 360
buffer = {col: deque(maxlen=BUFFER_SIZE) for col in LOG_COLS}

excel_backup_counter  = 0
baseline_save_counter = 0
latency_history       = deque(maxlen=500)


# =============================================================================
# SECTION 5 — DASHBOARD LAYOUT
# =============================================================================

app       = dash.Dash(__name__)
app.title = "TVOC Edge Deployment — GPO Kalman"

app.layout = html.Div([

    html.Div([
        html.H2(
            "Real-Time TVOC Monitoring \u2014 GPO-Tuned Kalman Filter",
            style={
                'textAlign': 'center', 'color': '#FFFFFF',
                'fontFamily': 'Georgia, serif', 'fontSize': '22px',
                'fontWeight': 'bold', 'marginBottom': '4px', 'marginTop': '10px',
            }
        ),
        html.P(
            id='status_bar',
            style={
                'textAlign': 'center', 'color': '#27AE60',
                'fontFamily': 'monospace', 'fontSize': '12px',
                'marginTop': '4px', 'marginBottom': '8px',
            }
        ),
    ], style={
        'backgroundColor': '#1a3a5c',
        'padding': '4px 12px 8px 12px',
        'borderBottom': '3px solid #0d2035',
    }),

    dcc.Interval(id='interval', interval=5000, n_intervals=0),

    html.Div([
        dcc.Graph(id='tvoc_panel', style={'height': '340px'},
                  config={'displayModeBar': False}),
    ], style={'margin': '10px 15px 0px 15px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='eco2_panel', style={'height': '230px'},
                      config={'displayModeBar': False}),
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            dcc.Graph(id='latency_panel', style={'height': '230px'},
                      config={'displayModeBar': False}),
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'margin': '0px 15px 10px 15px'}),

    html.Div([
        html.P(
            f"Q* = {Q_OPT}  |  R* = {R_OPT}  |  R*/Q* = {R_OPT/Q_OPT:.1f}  |  "
            f"Sat. threshold = {SAT_THRESHOLD:,} ppb  |  "
            f"Anomaly k = {ANOMALY_K}  |  Window = 30 min (360 samples)",
            style={
                'textAlign': 'center', 'color': '#AAAAAA',
                'fontFamily': 'monospace', 'fontSize': '11px', 'margin': '6px',
            }
        )
    ], style={'backgroundColor': '#1C1C1C', 'borderTop': '2px solid #333333'}),
])


# =============================================================================
# SECTION 6 — CALLBACK
# =============================================================================

@app.callback(
    Output('tvoc_panel',    'figure'),
    Output('eco2_panel',    'figure'),
    Output('latency_panel', 'figure'),
    Output('status_bar',    'children'),
    Input('interval',       'n_intervals')
)
def update_dashboard(n_intervals):
    global excel_backup_counter, baseline_save_counter

    # Read SGP30
    try:
        tvoc_raw = sgp30.TVOC
        eco2_raw = sgp30.eCO2
    except Exception as e:
        print(f"[ERROR] SGP30 read failed: {e}")
        tvoc_raw, eco2_raw = 0, 400

    # Read BME688 — uncomment when connected
    # try:
    #     temperature = bme688.temperature
    #     humidity    = bme688.relative_humidity
    # except Exception:
    #     temperature = float('nan')
    #     humidity    = float('nan')
    temperature = float('nan')
    humidity    = float('nan')

    # Apply filter
    tvoc_filt, K_gain, P_err, z_used, sat_flag, anomaly, latency_ms = \
        kalman_update(tvoc_raw)
    innov_approx = float(tvoc_raw) - tvoc_filt

    latency_history.append(latency_ms)
    timestamp = datetime.now()

    # Build log row
    row = {
        'Timestamp'    : timestamp,
        'TVOC_raw'     : tvoc_raw,
        'TVOC_filtered': round(tvoc_filt, 3),
        'eCO2_raw'     : eco2_raw,
        'Temperature_C': round(temperature, 4) if not pd.isna(temperature) else None,
        'Humidity_pct' : round(humidity,    4) if not pd.isna(humidity)    else None,
        'Kalman_gain'  : round(K_gain,  6),
        'P_error'      : round(P_err,   6),
        'Innovation'   : round(innov_approx, 3),
        'Sat_flag'     : sat_flag,
        'Anomaly_flag' : anomaly,
        'Latency_ms'   : round(latency_ms, 4),
    }
    for col, val in row.items():
        buffer[col].append(val)

    # Append to CSV
    with open(LOG_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([row[c] for c in LOG_COLS])

    # Excel backup
    excel_backup_counter += 1
    if excel_backup_counter >= EXCEL_BACKUP_EVERY:
        try:
            pd.read_csv(LOG_CSV).to_excel(LOG_EXCEL, index=False)
            print(f"[{timestamp.strftime('%H:%M:%S')}] Excel backup written")
        except Exception as e:
            print(f"[WARNING] Excel backup failed: {e}")
        excel_backup_counter = 0

    # Baseline save
    baseline_save_counter += 1
    if baseline_save_counter >= BASELINE_SAVE_EVERY:
        save_baseline()
        baseline_save_counter = 0

    # Build DataFrame from ring buffer
    df = pd.DataFrame({col: list(buffer[col]) for col in LOG_COLS})

    # Panel 1: TVOC + Kalman gain
    fig_tvoc = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=["TVOC (ppb)", "Kalman Gain K"]
    )

    sigma_band = df['P_error'].values ** 0.5
    fig_tvoc.add_trace(go.Scatter(
        x=pd.concat([df['Timestamp'], df['Timestamp'][::-1]]),
        y=list(df['TVOC_filtered'] + sigma_band) +
          list((df['TVOC_filtered'] - sigma_band)[::-1]),
        fill='toself', fillcolor='rgba(192,57,43,0.13)',
        line=dict(color='rgba(0,0,0,0)'),
        name='\u00B11\u03C3 uncertainty', showlegend=True
    ), row=1, col=1)

    fig_tvoc.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['TVOC_raw'],
        mode='lines', name='Raw TVOC',
        line=dict(color='#AAAAAA', width=1.2, dash='dot'), opacity=0.7
    ), row=1, col=1)

    fig_tvoc.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['TVOC_filtered'],
        mode='lines', name='GPO Kalman filtered',
        line=dict(color='#C0392B', width=2.5)
    ), row=1, col=1)

    sat_df = df[df['Sat_flag'] == True]
    if len(sat_df) > 0:
        fig_tvoc.add_trace(go.Scatter(
            x=sat_df['Timestamp'], y=sat_df['TVOC_raw'],
            mode='markers', name='Saturation artifact',
            marker=dict(color='#E67E22', size=9, symbol='x-thin-open',
                        line=dict(width=2))
        ), row=1, col=1)

    anom_df = df[df['Anomaly_flag'] == True]
    if len(anom_df) > 0:
        fig_tvoc.add_trace(go.Scatter(
            x=anom_df['Timestamp'], y=anom_df['TVOC_raw'],
            mode='markers', name=f'Anomaly (k={ANOMALY_K})',
            marker=dict(color='#F39C12', size=8, symbol='triangle-down',
                        line=dict(width=1, color='#E67E22'))
        ), row=1, col=1)

    fig_tvoc.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['Kalman_gain'],
        mode='lines', name='Kalman gain K',
        line=dict(color='#2980B9', width=2.0)
    ), row=2, col=1)

    fig_tvoc.update_layout(
        title=dict(
            text=(
                f"Panel 1.  TVOC Signal  [Q*={Q_OPT}, R*={R_OPT}]  \u2502  "
                f"raw={tvoc_raw} ppb  filtered={tvoc_filt:.1f} ppb  "
                f"K={K_gain:.4f}  \u03C3={P_err**0.5:.3f}"
            ),
            font=dict(size=13, family='Georgia, serif'), x=0.5, xanchor='center'
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=11),
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#CCCCCC', borderwidth=1),
        template='plotly_white',
        margin=dict(l=65, r=20, t=55, b=20),
        hovermode='x unified',
        paper_bgcolor='white', plot_bgcolor='#FAFBFC',
    )
    fig_tvoc.update_yaxes(title_text="TVOC (ppb)", title_font_size=12, row=1, col=1)
    fig_tvoc.update_yaxes(title_text="K", title_font_size=12,
                          range=[0, 1], row=2, col=1)
    fig_tvoc.update_xaxes(title_text="Time", title_font_size=12, row=2, col=1)

    # Panel 2: eCO2 auxiliary channel
    fig_eco2 = go.Figure()
    fig_eco2.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['eCO2_raw'],
        mode='lines', name='eCO\u2082 (ppm)',
        line=dict(color='#16A085', width=2.2)
    ))
    fig_eco2.update_layout(
        title=dict(
            text=f"Panel 2.  eCO\u2082 Auxiliary Channel  \u2502  Latest: {eco2_raw} ppm",
            font=dict(size=12, family='Georgia, serif'), x=0.5, xanchor='center'
        ),
        xaxis_title="Time", yaxis_title="eCO\u2082 (ppm)",
        yaxis_title_font=dict(size=12),
        template='plotly_white',
        margin=dict(l=65, r=20, t=50, b=45),
        legend=dict(font=dict(size=11)),
        paper_bgcolor='white', plot_bgcolor='#FAFBFC',
    )

    # Panel 3: Latency histogram
    lat_vals = list(latency_history)
    mean_lat = sum(lat_vals) / len(lat_vals) if lat_vals else 0.0
    fig_lat  = go.Figure()
    fig_lat.add_trace(go.Histogram(
        x=lat_vals, nbinsx=30,
        marker_color='#8E44AD', opacity=0.85, name='Latency (ms)'
    ))
    fig_lat.add_vline(
        x=mean_lat, line_dash='dash', line_color='#E74C3C', line_width=2,
        annotation_text=f'mean = {mean_lat:.4f} ms',
        annotation_position='top right', annotation_font_size=11
    )
    fig_lat.update_layout(
        title=dict(
            text=(
                f"Panel 3.  Filter Arithmetic Latency  \u2502  "
                f"Mean = {mean_lat:.4f} ms  "
                f"({mean_lat/5000*100:.4f}% of 5,000 ms budget)"
            ),
            font=dict(size=12, family='Georgia, serif'), x=0.5, xanchor='center'
        ),
        xaxis_title="Latency (ms)", yaxis_title="Count",
        xaxis_title_font=dict(size=12), yaxis_title_font=dict(size=12),
        template='plotly_white',
        margin=dict(l=65, r=20, t=50, b=45),
        showlegend=False,
        paper_bgcolor='white', plot_bgcolor='#FAFBFC',
    )

    # Status bar
    status = (
        f"{timestamp.strftime('%Y-%m-%d  %H:%M:%S')}  \u2502  "
        f"TVOC raw={tvoc_raw} ppb  filtered={tvoc_filt:.1f} ppb  "
        f"K={K_gain:.4f}  \u03C3={P_err**0.5:.3f}  \u2502  "
        f"eCO\u2082={eco2_raw} ppm  \u2502  "
        f"Sat.={n_sat}  \u2502  Anomaly={n_anomaly}  \u2502  "
        f"Latency={latency_ms:.3f} ms  \u2502  Samples={n_intervals+1:,}"
    )

    return fig_tvoc, fig_eco2, fig_lat, status


# =============================================================================
# SECTION 7 — RUN
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  TVOC Real-Time Edge Deployment")
    print("=" * 55)
    print(f"  Q*             = {Q_OPT}")
    print(f"  R*             = {R_OPT}")
    print(f"  R*/Q*          = {R_OPT/Q_OPT:.1f}")
    print(f"  Anomaly k      = {ANOMALY_K}")
    print(f"  Sat. threshold = {SAT_THRESHOLD:,} ppb")
    print(f"  Log (CSV)      = {LOG_CSV}")
    print(f"  Log (Excel)    = {LOG_EXCEL}")
    print(f"  Baseline file  = {BASELINE_FILE}")
    print(f"  Dashboard      = http://<raspberry-pi-ip>:8050")
    print("=" * 55)
    print()
    app.run(host='0.0.0.0', port=8050, debug=False)
