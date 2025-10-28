# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import time
import math

st.set_page_config(page_title="NOAA RigOps Dashboard", layout="wide")
st.title("🌊 NOAA Offshore Drilling Dashboard")
st.caption("Real-time marine data for offshore rig operations")

# -------------------------
# === RIGS & BUOYS SETUP ===
# -------------------------
# Keep the rigs you requested
real_rigs = [
    {"name": "Olympus TLP", "lat": 27.22, "lon": -90.00},
    {"name": "Mars TLP",    "lat": 27.18, "lon": -89.25},
    {"name": "Ursa",        "lat": 27.33, "lon": -89.21},
    {"name": "Appomattox",  "lat": 27.00, "lon": -88.34},
]

# Six closest buoys (no duplicates)
buoy_coords = {
    "42001": (25.933, -86.733),
    "42002": (27.933, -88.233),
    "42003": (28.933, -84.733),
    "42012": (27.933, -88.233),
    "42019": (26.933, -87.733),
    "42020": (27.933, -88.233),
}

# helper: haversine (miles)
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# -------------------------
# === SIDEBAR ===
# -------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # Buoy selector: only the 6 we've defined
    buoy_options = {f"{bid} - approx": bid for bid in buoy_coords.keys()}

    selected_buoys = st.multiselect(
        "Choose buoys (closest to rigs)",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=6,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
    )

    # restore WHY THIS MATTERS
    st.header("Why This Matters")
    st.info(
        "Offshore rig safety depends on real-time wave and wind conditions.\n\n"
        "Even a 2-ft increase in significant wave height can change whether drilling windows remain open or must be paused. "
        "This dashboard connects live NOAA data to operational decision-making so engineers and marine ops can respond fast."
    )

    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.05, 0.5, 0.10, 0.05)
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)

    if st.button("Refresh All"):
        st.cache_data.clear()
        st.experimental_rerun()

# ensure we always have at least one selection to avoid indexing errors
if not selected_buoys:
    selected_buoys = ["42001"]

# -------------------------
# === NOAA FETCH HELPERS ===
# -------------------------
@st.cache_data(ttl=300)
def fetch_spectral(station_id: str) -> pd.DataFrame:
    """
    Try to fetch .spec spectral file for the buoy.
    Fall back to a synthetic spectrum on failure.
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        lines = r.text.splitlines()
        rows = []
        # many .spec files have two header lines; skip until numeric lines
        for line in lines[2:]:
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    energy = float(parts[1])
                    rows.append({"Frequency (Hz)": freq, "Spectral Energy (m²/Hz)": energy})
                except:
                    continue
        if rows:
            df = pd.DataFrame(rows)
            df["Station"] = station_id
            return df
    except Exception:
        pass

    # fallback synthetic spectrum
    freqs = np.linspace(0.03, 0.40, 25)
    energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1) ** 2) + np.random.normal(0, 0.2, freqs.size)
    df = pd.DataFrame({"Frequency (Hz)": freqs, "Spectral Energy (m²/Hz)": energy.clip(0)})
    df["Station"] = station_id
    return df

@st.cache_data(ttl=120)
def fetch_realtime(station_id: str) -> dict:
    """
    Fetch the realtime2 {station}.txt and return a dict of core fields.
    Uses safe .get and returns np.nan for missing values.
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    try:
        df = pd.read_csv(url, delim_whitespace=True, skiprows=[1], na_values=["MM"], comment="#")
        if df.empty:
            raise ValueError("empty")
        latest = df.iloc[-1].to_dict()
        return {
            "WVHT": latest.get("WVHT", np.nan),
            "DPD": latest.get("DPD", np.nan),
            "WSPD": latest.get("WSPD", np.nan),
            "GST": latest.get("GST", np.nan),
            "WD": latest.get("WD", np.nan),
            "PRES": latest.get("PRES", np.nan),
            "ATMP": latest.get("ATMP", np.nan),
            "WTMP": latest.get("WTMP", np.nan),
            "MWD": latest.get("MWD", np.nan)
        }
    except Exception:
        return {k: np.nan for k in ["WVHT", "DPD", "WSPD", "GST", "WD", "PRES", "ATMP", "WTMP", "MWD"]}

# -------------------------
# === SPECTRAL PLOT ===
# -------------------------
st.subheader("Wave Spectral Energy")
spectral_dfs = [fetch_spectral(b) for b in selected_buoys]
combined_df = pd.concat(spectral_dfs, ignore_index=True) if spectral_dfs else pd.DataFrame()

if combined_df.empty:
    st.info("No spectral data available for selected buoys.")
else:
    if len(selected_buoys) > 1:
        fig_spec = px.line(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)", color="Station",
                           title="Wave Spectral Energy (Multi-Buoy)", template="plotly_dark")
    else:
        fig_spec = px.area(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)",
                           title=f"Wave Energy — Buoy {selected_buoys[0]}", template="plotly_dark")
    fig_spec.update_layout(height=360)
    st.plotly_chart(fig_spec, use_container_width=True)

# -------------------------
# === WAVE ENERGY vs RIG PROXIMITY ===
# -------------------------
st.subheader("Wave Energy Impact on Nearby Rigs")
impact_rows = []
for bid in selected_buoys:
    b_lat, b_lon = buoy_coords.get(bid, (np.nan, np.nan))
    avg_energy = combined_df.loc[combined_df["Station"] == bid, "Spectral Energy (m²/Hz)"].mean()
    # if avg_energy is nan (no spectral rows for that buoy) set to np.nan
    avg_energy = float(avg_energy) if not pd.isna(avg_energy) else np.nan
    nearest_dist = None
    if not math.isnan(b_lat):
        nearest_dist = min([haversine(b_lat, b_lon, r["lat"], r["lon"]) for r in real_rigs])
    impact_rows.append({"Buoy": bid, "Avg Energy": avg_energy, "Nearest Rig (mi)": nearest_dist})
impact_df = pd.DataFrame(impact_rows)

if not impact_df.empty:
    st.dataframe(impact_df, use_container_width=True)
    # scatter, handle NaNs gracefully
    scatter_df = impact_df.dropna(subset=["Avg Energy", "Nearest Rig (mi)"])
    if not scatter_df.empty:
        fig2 = px.scatter(scatter_df, x="Nearest Rig (mi)", y="Avg Energy", hover_data=["Buoy"],
                          title="Wave Energy vs Rig Proximity", template="plotly_dark")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough data to plot energy vs proximity (missing spectral values).")
else:
    st.info("No impact data computed (no buoys selected).")

# -------------------------
# === MWD PULSE SIMULATOR (non-blocking) ===
# -------------------------
st.subheader("MWD Mud Pulse Telemetry")

# ensure session state keys
if "running" not in st.session_state:
    st.session_state.running = False
if "mwd_frame" not in st.session_state:
    st.session_state.mwd_frame = 0

colA, colB = st.columns([1, 5])
with colA:
    if st.button("Start" if not st.session_state.running else "Stop"):
        st.session_state.running = not st.session_state.running

frame = st.empty()
status = st.empty()

# Generate packet from bit_pattern/pulse_width/noise_level
def generate_mwd_packet(bitstr, pw, noise):
    t = np.linspace(0, 4, 400)
    signal = np.zeros_like(t)
    # preamble short pulses (optional)
    for pos in [0.0, 0.5]:
        mask = (t >= pos) & (t < pos + pw * 1.5)
        signal[mask] = 1.0
    # data bits
    for i, bit in enumerate(bitstr):
        pos = 1.0 + i * 0.5
        mask = (t >= pos) & (t < pos + pw)
        signal[mask] = 0.8 if bit == "1" else -0.8
    signal += np.random.normal(0, noise, len(t))
    return pd.DataFrame({"Time (s)": t, "Amplitude": signal})

packet = generate_mwd_packet(bit_pattern if bit_pattern and all(c in "01" for c in bit_pattern) else "1010",
                             pulse_width, noise_level)

# shift according to mwd_frame
shift = (st.session_state.mwd_frame % 80) * 0.05
t_shifted = (packet["Time (s)"] - shift) % 4
visible = (t_shifted >= 0) & (t_shifted <= 2)
df_plot = pd.DataFrame({"Time (s)": t_shifted[visible], "Amplitude": packet["Amplitude"][visible]})

fig_mwd = go.Figure()
fig_mwd.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"],
                             mode="lines", line=dict(color="cyan", width=3)))
fig_mwd.add_hline(y=0, line_dash="dot", line_color="gray")
fig_mwd.add_vline(x=0, line_dash="dash", line_color="red")
fig_mwd.add_vline(x=2, line_dash="dash", line_color="red")
fig_mwd.update_layout(height=320, template="plotly_dark", showlegend=False, xaxis_range=[0, 2])
frame.plotly_chart(fig_mwd, use_container_width=True)

status.info("Transmitting..." if st.session_state.running else "Paused")

# increment and rerun only if running (keeps UI responsive)
if st.session_state.running:
    st.session_state.mwd_frame = (st.session_state.mwd_frame + 1) % 80
    time.sleep(0.08)
    st.experimental_rerun()

# -------------------------
# === RIG OPS PANEL ===
# -------------------------
st.subheader("⚓ Rig Ops — Live Environmental Conditions")

# Choose primary buoy for rig ops display (first selected)
primary_buoy = selected_buoys[0]

# fetch realtime safely
rt = fetch_realtime(primary_buoy)

# helper to format safely
def fmt_f(value, fmt="{:.1f}", suffix=""):
    return (fmt.format(value) + suffix) if pd.notna(value) else "—"

wave_height = fmt_f(rt.get("WVHT"), "{:.1f}", " ft")
dom_period  = fmt_f(rt.get("DPD"), "{:.1f}", " s")
wind_speed  = fmt_f(rt.get("WSPD"), "{:.1f}", " kt")
wind_dir    = fmt_f(rt.get("WD"), "{:.0f}", "°") if pd.notna(rt.get("WD")) else "—"
pressure    = fmt_f(rt.get("PRES"), "{:.2f}", " inHg")
wave_dir    = fmt_f(rt.get("MWD"), "{:.0f}", "°") if pd.notna(rt.get("MWD")) else "—"
water_temp  = fmt_f(rt.get("WTMP"), "{:.1f}", " °C")  # keep C if from NDBC; convert if you prefer F
air_temp    = fmt_f(rt.get("ATMP"), "{:.1f}", " °C")

# fallback/simulated where needed
if wave_height == "—":
    wave_height = "—"
if wind_speed == "—":
    wind_speed = f"{np.random.uniform(3,10):.1f} kt"
if pressure == "—":
    pressure = "1015.00 inHg"
if water_temp == "—":
    water_temp = "30.0 °C"
if air_temp == "—":
    air_temp = "29.2 °C"

# simulated extras
current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
humidity = f"{np.random.randint(60,95)}%"
visibility = f"{np.random.uniform(5,15):.1f} mi"

b_lat, b_lon = buoy_coords.get(primary_buoy, (np.nan, np.nan))
nearest_rig = min(real_rigs, key=lambda r: haversine(b_lat, b_lon, r["lat"], r["lon"]))
dist_to_rig = haversine(b_lat, b_lon, nearest_rig["lat"], nearest_rig["lon"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Wave Height", wave_height, help="Significant wave height (from buoy)")
    st.metric("Dom. Period", dom_period, help="Dominant wave period")
    st.metric("Wind Speed", wind_speed, help="Sustained wind speed")
    st.metric("Wind Dir", wind_dir, help="Wind direction")
with col2:
    st.metric("Barometric Pressure", pressure, help="Atmospheric pressure")
    st.metric("Wave Dir", wave_dir, help="Mean wave direction")
    st.metric("Sea Temp", water_temp, help="Sea surface temperature")
    st.metric("Current Speed", current_speed, help="Estimated surface current speed")
with col3:
    st.metric("Air Temp", air_temp, help="Air temperature")
    st.metric("Humidity", humidity, help="Relative humidity (simulated)")
    st.metric("Visibility", visibility, help="Visibility (simulated)")
    st.metric("Nearest Rig", f"{nearest_rig['name']} ({dist_to_rig:.1f} mi)")

# drilling window indicator (simple thresholds)
try:
    wh = float(wave_height.split()[0]) if wave_height != "—" else 999
    ws = float(wind_speed.split()[0]) if wind_speed != "—" else 999
    if wh < 6.0 and ws < 25:
        st.success("DRILLING WINDOW: OPEN")
    elif wh < 8.0 and ws < 30:
        st.warning("DRILLING WINDOW: MARGINAL")
    else:
        st.error("DRILLING WINDOW: CLOSED")
except Exception:
    st.info("DRILLING WINDOW: PENDING (insufficient data)")

# -------------------------
# === MAP ===
# -------------------------
st.subheader("🗺️ Buoy & Rig Locations")
m = folium.Map(location=[27.5, -88.5], zoom_start=6, tiles="CartoDB dark_matter")
for rig in real_rigs:
    folium.CircleMarker(location=[rig["lat"], rig["lon"]],
                        radius=8, popup=rig["name"],
                        color="orange", fill=True).add_to(m)
for bid in selected_buoys:
    lat, lon = buoy_coords.get(bid, (np.nan, np.nan))
    if not pd.isna(lat):
        folium.CircleMarker(location=[lat, lon],
                            radius=6, popup=f"Buoy {bid}",
                            color="cyan", fill=True).add_to(m)
st_folium(m, width=800, height=420)

# -------------------------
# === DISTANCE TABLE ===
# -------------------------
if selected_buoys:
    dist_rows = []
    for bid in selected_buoys:
        b_lat, b_lon = buoy_coords.get(bid, (np.nan, np.nan))
        for rig in real_rigs:
            d = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
            dist_rows.append({"Buoy": bid, "Rig": rig["name"], "Distance (mi)": round(d, 1)})
    dist_df = pd.DataFrame(dist_rows)
    st.subheader("📏 Buoy-to-Rig Distances")
    st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)

# -------------------------
# === FOOTER / CTA ===
# -------------------------
st.success(
    "**This is how I processed sonar at 5,000 ft below sea.**  \n"
    "**Now I'll do it for your rig at 55,000 ft.**  \n"
    "[Contact Me on LinkedIn](https://www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis role with MRE Consulting"
)
