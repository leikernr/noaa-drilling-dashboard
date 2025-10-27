# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="Gulf Rig Ops", layout="wide")
st.title("Submarine Sonar to Subsea Sensors — Gulf of Mexico Fleet")
st.markdown("""
**Real-Time Acoustic Energy + Rig Operations Dashboard**  
*From U.S. Navy STS2 sonar processing to MWD drilling optimization*  
Built by a submarine veteran with 14 years in oilfield telemetry
""")

# ---------------------
# === CONFIG / DATA ===
# ---------------------

# 6 chosen buoys (closest coverage for your rigs, no duplicates)
buoy_coords = {
    "42001": [25.933, -86.733],  # West Florida Basin
    "42002": [27.933, -88.233],  # Central Gulf
    "42003": [28.933, -84.733],  # East Gulf
    "42012": [27.933, -88.233],  # Central Gulf Platform (support)
    "42019": [26.933, -87.733],  # West Florida Shelf
    "42020": [27.933, -88.233],  # Central Gulf (support)
}

# Rigs (example fleet)
real_rigs = [
    {"name": "Neptune TLP", "lat": 27.37, "lon": -89.92},
    {"name": "Thunder Hawk SPAR", "lat": 28.18, "lon": -88.67},
    {"name": "King's Quay FPS", "lat": 27.75, "lon": -89.25},
    {"name": "Sailfin FPSO", "lat": 27.80, "lon": -90.20}
]

# Haversine (miles)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    km = R * c
    return km * 0.621371  # miles

# Sidebar: select up to 6 (we default to all six)
with st.sidebar:
    st.header("Gulf Buoy Fleet (Select Up to 6)")
    buoy_options = {f"{k} — {v}": k for k, v in {
        "42001": "West Florida Basin",
        "42002": "Central Gulf",
        "42003": "East Gulf",
        "42012": "Central Gulf Platform",
        "42019": "West Florida Shelf",
        "42020": "Central Gulf (support)"
    }.items()}
    # Show labels but store IDs
    selected_buoys = st.multiselect(
        "Choose buoys",
        options=list(buoy_options.keys()),
        default=list(buoy_options.keys()),
        max_selections=6
    )
    # convert back to ids
    selected_buoys = [buoy_options[label] for label in selected_buoys]

    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = real-time signal processing  
    - **MWD Drilling** = same math: gamma, resistivity, torque  
    - **Energy Tech** = multi-source fusion for ops  
    """)
    st.info("NOAA 420xx → Gulf Fleet → Multi-rig sensor analogy")

    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.05, 0.5, 0.1, 0.05)
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)

    if st.button("Refresh All"):
        st.cache_data.clear()
        st.success("Refreshed caches")

# -------------------------
# === NOAA DATA INGEST ===
# -------------------------
# Cache live calls for short TTL to reduce load
@st.cache_data(ttl=300)
def get_noaa_spectrum(station_id):
    """
    Try to fetch a .spec spectral file from NDBC realtime2.
    Fallback: simulated gaussian-like spectrum.
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        lines = r.text.splitlines()
        data = []
        # skip header lines until data starts (common NDBC .spec has two header lines)
        for line in lines[2:]:
            if line.strip() and not line.startswith('#'):
                cols = line.split()
                if len(cols) >= 2:
                    try:
                        freq = float(cols[0])
                        energy = float(cols[1])
                        data.append({"Frequency (Hz)": freq, "Spectral Energy (m²/Hz)": energy})
                    except:
                        continue
        if data:
            df = pd.DataFrame(data)
            df["Station"] = station_id
            return df
    except Exception:
        pass

    # fallback simulated spectrum
    freqs = np.linspace(0.03, 0.40, 25)
    energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1)**2) + np.random.normal(0, 0.2, freqs.size)
    return pd.DataFrame({
        "Frequency (Hz)": freqs,
        "Spectral Energy (m²/Hz)": energy.clip(0),
        "Station": [station_id] * freqs.size
    })

@st.cache_data(ttl=120)
def get_realtime_buoy_data(station_id):
    """
    Fetch the latest .txt file from NDBC realtime2.
    Returns a dict with key operational fields. If unavailable, returns NaNs.
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    try:
        # NDBC text files: first line header, second line units -> skip second line
        df = pd.read_csv(url, delim_whitespace=True, skiprows=[1], comment='#', na_values=['MM'])
        if df.empty:
            raise ValueError("Empty dataframe")
        latest = df.iloc[-1].to_dict()
        # common column names: WVHT, DPD, WSPD, GST, WD, PRES, ATMP, WTMP, MWD
        out = {
            "WVHT": latest.get('WVHT', np.nan),
            "DPD": latest.get('DPD', np.nan),
            "WSPD": latest.get('WSPD', np.nan),
            "GST": latest.get('GST', np.nan),
            "WD": latest.get('WD', np.nan),
            "PRES_mb": latest.get('PRES', np.nan),   # NDBC expresses pressure in mb (hPa)
            "ATMP_C": latest.get('ATMP', np.nan),
            "WTMP_C": latest.get('WTMP', np.nan),
            "MWD": latest.get('MWD', np.nan)
        }
        # Convert PRES (mb) -> inHg if available
        if not pd.isna(out["PRES_mb"]):
            out["PRES_inHg"] = out["PRES_mb"] * 0.0295299830714
        else:
            out["PRES_inHg"] = np.nan
        return out
    except Exception:
        # fallback: simulated/no-data
        return {k: np.nan for k in ["WVHT", "DPD", "WSPD", "GST", "WD", "PRES_mb", "ATMP_C", "WTMP_C", "MWD", "PRES_inHg"]}

# -------------------------
# === COMBINE SPECTRA ===
# -------------------------
# load spectral dfs for selected buoys
spectral_dfs = []
for b in selected_buoys:
    spectral_dfs.append(get_noaa_spectrum(b))
combined_df = pd.concat(spectral_dfs, ignore_index=True) if spectral_dfs else pd.DataFrame()

# === SPECTRAL PLOT ===
st.subheader("Wave Spectral Energy")
if not combined_df.empty:
    if len(selected_buoys) > 1:
        fig1 = px.line(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)", color="Station",
                       title="Wave Spectral Energy (Multi-Buoy)", template="plotly_dark")
    else:
        fig1 = px.area(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)",
                       title=f"Wave Energy — Buoy {selected_buoys[0]}", template="plotly_dark")
    fig1.update_layout(height=350)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Select buoys to view wave energy.")

# -------------------------
# === MWD SIMULATOR ===
# -------------------------
st.subheader("MWD Mud Pulse Telemetry")
if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
    bit_pattern = "1010"

# control start/stop
if 'running' not in st.session_state:
    st.session_state.running = True

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Stop" if st.session_state.running else "Start"):
        st.session_state.running = not st.session_state.running

frame = st.empty()
status = st.empty()

def generate_mwd_packet():
    t = np.linspace(0, 4, 400)
    signal = np.zeros_like(t)
    # short preamble pulses
    for pos in [0.0, 0.5]:
        mask = (t >= pos) & (t < pos + pulse_width * 1.5)
        signal[mask] = 1.0
    # data bits starting at t=1.0
    for i, bit in enumerate(bit_pattern):
        pos = 1.0 + i * 0.5
        mask = (t >= pos) & (t < pos + pulse_width)
        signal[mask] = 0.8 if bit == '1' else -0.8
    # noise
    signal += np.random.normal(0, noise_level, len(t))
    return pd.DataFrame({"Time (s)": t, "Amplitude": signal})

if st.session_state.running:
    packet = generate_mwd_packet()
    # simple animation loop (limited iterations so Streamlit doesn't hang)
    for shift in np.linspace(0, 4, 80):
        if not st.session_state.running:
            break
        t_shifted = (packet["Time (s)"] - shift) % 4
        visible = (t_shifted >= 0) & (t_shifted <= 2)
        df_plot = pd.DataFrame({"Time (s)": t_shifted[visible], "Amplitude": packet["Amplitude"][visible]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"],
                                 mode='lines', line=dict(color='cyan', width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(x=2, line_dash="dash", line_color="red")
        fig.update_layout(height=300, template="plotly_dark", showlegend=False, xaxis_range=[0,2])
        frame.plotly_chart(fig, use_container_width=True)
        status.info("Transmitting...")
        time.sleep(0.08)
else:
    packet = generate_mwd_packet()
    df_plot = packet[packet["Time (s)"] <= 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"],
                             mode='lines', line=dict(color='cyan', width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(x=2, line_dash="dash", line_color="red")
    fig.update_layout(height=300, template="plotly_dark", showlegend=False, xaxis_range=[0,2])
    frame.plotly_chart(fig, use_container_width=True)
    status.info("Paused.")

# -------------------------
# === RIG OPS PANEL ===
# -------------------------
st.subheader("Rig Ops — Live Environmental Conditions")

if not selected_buoys:
    st.info("Select a buoy to view live conditions.")
else:
    buoy = selected_buoys[0]
    data = get_realtime_buoy_data(buoy)

    # format values with fallbacks
    wave_height = f"{data['WVHT']:.1f} ft" if not pd.isna(data['WVHT']) else "—"
    dom_period = f"{data['DPD']:.1f} s" if not pd.isna(data['DPD']) else "—"
    wind_speed = f"{data['WSPD']:.1f} kt" if not pd.isna(data['WSPD']) else "—"
    wind_dir = f"{int(data['WD'])}°" if not pd.isna(data['WD']) else "—"
    pres = f"{data['PRES_inHg']:.2f} inHg" if not pd.isna(data['PRES_inHg']) else "—"
    wave_dir = f"{int(data['MWD'])}°" if not pd.isna(data['MWD']) else "—"
    water_temp = f"{(data['WTMP_C'] * 9/5 + 32):.1f}°F" if not pd.isna(data['WTMP_C']) else "—"
    air_temp = f"{(data['ATMP_C'] * 9/5 + 32):.1f}°F" if not pd.isna(data['ATMP_C']) else "—"

    # simulated values when not provided by NDBC
    current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
    current_dir = f"{np.random.randint(0, 360)}°"
    humidity = f"{np.random.randint(60, 95)}%"
    visibility = f"{np.random.uniform(5, 15):.1f} mi"
    subsurface_temp = (f"{(data['WTMP_C'] * 9/5 + 32 - 5):.1f}°F" if not pd.isna(data['WTMP_C']) else "—")
    salinity = "35.0 PSU"

    # nearest rig
    buoy_lat, buoy_lon = buoy_coords[buoy]
    nearest_rig = min(real_rigs, key=lambda r: haversine(buoy_lat, buoy_lon, r["lat"], r["lon"]))
    dist = haversine(buoy_lat, buoy_lon, nearest_rig["lat"], nearest_rig["lon"])

    # show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Wave Height", wave_height, help="Rig motion, BHA run")
        st.metric("Dom. Period", dom_period, help="Wave type (swell/chop)")
        st.metric("Wind Speed", wind_speed, help="DP, crane ops")
        st.metric("Wind Dir", wind_dir, help="Crane swing")
    with col2:
        st.metric("Barometric Pressure", pres, help="Storm forecast")
        st.metric("Wave Dir", wave_dir, help="Vessel heading")
        st.metric("Sea Temp", water_temp, help="Mud cooling, ROV")
        st.metric("Current Speed", current_speed, help="Riser stress")
    with col3:
        st.metric("Air Temp", air_temp, help="Crew comfort")
        st.metric("Humidity", humidity, help="Fog risk")
        st.metric("Visibility", visibility, help="Helicopter ops")
        st.metric("Nearest Rig", nearest_rig["name"], f"{dist:.0f} mi")

    # drilling window logic
    try:
        wh = float(wave_height.split()[0]) if wave_height != "—" else 99
        ws = float(wind_speed.split()[0]) if wind_speed != "—" else 99
        if wh < 6.0 and ws < 25:
            st.success("DRILLING WINDOW: OPEN")
        elif wh < 8.0 and ws < 30:
            st.warning("DRILLING WINDOW: MARGINAL")
        else:
            st.error("DRILLING WINDOW: CLOSED")
    except:
        st.info("DRILLING WINDOW: PENDING")

# -------------------------
# === WAVE ENERGY vs RIG PROXIMITY (ENHANCED) ===
# -------------------------
st.subheader("Wave & Environmental Impact on Nearby Rigs")
st.markdown("""
These metrics show **how offshore energy conditions propagate through the Gulf**  
and **impact operational readiness at each rig**.  
Each row is a NOAA buoy, scaled by **wave height** and colored by **wind speed**.
""")

if selected_buoys:
    # average spectral energy per buoy
    if not combined_df.empty:
        energy_summary = combined_df.groupby("Station")["Spectral Energy (m²/Hz)"].mean().reset_index()
        energy_summary.columns = ["Buoy", "Avg Wave Energy (m²/Hz)"]
    else:
        energy_summary = pd.DataFrame({"Buoy": selected_buoys, "Avg Wave Energy (m²/Hz)": [np.nan]*len(selected_buoys)})

    impact_rows = []
    for buoy_id in selected_buoys:
        real = get_realtime_buoy_data(buoy_id)
        if buoy_id in buoy_coords:
            b_lat, b_lon = buoy_coords[buoy_id]
            nearest_rig_obj = min(real_rigs, key=lambda r: haversine(b_lat, b_lon, r["lat"], r["lon"]))
            dist_mi = haversine(b_lat, b_lon, nearest_rig_obj["lat"], nearest_rig_obj["lon"])
            energy_row = energy_summary[energy_summary["Buoy"] == buoy_id]
            avg_energy = float(energy_row["Avg Wave Energy (m²/Hz)"]) if not energy_row.empty else np.nan

            impact_rows.append({
                "Buoy": buoy_id,
                "Nearest Rig": nearest_rig_obj["name"],
                "Distance (mi)": round(dist_mi, 1),
                "Avg Wave Energy (m²/Hz)": round(avg_energy, 2) if not pd.isna(avg_energy) else np.nan,
                "Wave Height (ft)": real.get("WVHT", np.nan),
                "Wind Speed (kt)": real.get("WSPD", np.nan),
                "Pressure (inHg)": round(real.get("PRES_inHg", np.nan), 2) if not pd.isna(real.get("PRES_inHg", np.nan)) else np.nan,
                "Sea Temp (°F)": round((real["WTMP_C"] * 9/5 + 32), 1) if not pd.isna(real.get("WTMP_C")) else np.nan,
                "Wave Dir (°)": real.get("MWD", np.nan)
            })

    impact_df = pd.DataFrame(impact_rows)

    if not impact_df.empty:
        st.dataframe(impact_df.style.background_gradient(subset=["Avg Wave Energy (m²/Hz)"], cmap="Blues"), use_container_width=True)

        fig2 = px.scatter(
            impact_df,
            x="Distance (mi)",
            y="Avg Wave Energy (m²/Hz)",
            color="Wind Speed (kt)",
            size="Wave Height (ft)",
            hover_data=["Buoy", "Nearest Rig", "Sea Temp (°F)", "Pressure (inHg)"],
            title="Environmental Energy vs Rig Distance — Operational Impact",
            template="plotly_dark"
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No valid buoy data available.")
else:
    st.info("Select buoys to view environmental impact.")

# -------------------------
# === MAP (Buoys + Rigs) ===
# -------------------------
st.subheader("Buoy & Rig Locations")
m = folium.Map(location=[27.5, -88.5], zoom_start=6, tiles="CartoDB dark_matter")

for bid, (lat, lon) in buoy_coords.items():
    if bid in selected_buoys:
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=f"Buoy {bid}",
            color="cyan",
            fill=True
        ).add_to(m)

for rig in real_rigs:
    folium.CircleMarker(
        location=[rig["lat"], rig["lon"]],
        radius=10,
        popup=rig["name"],
        color="orange",
        fill=True
    ).add_to(m)

st_folium(m, width=800, height=420)

# -------------------------
# === BUOY-TO-RIG DISTANCES TABLE ===
# -------------------------
if selected_buoys:
    st.subheader("Buoy-to-Rig Distances")
    dist_rows = []
    for bid in selected_buoys:
        if bid in buoy_coords:
            b_lat, b_lon = buoy_coords[bid]
            for rig in real_rigs:
                dist = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
                dist_rows.append({"Buoy": bid, "Rig": rig["name"], "Distance (mi)": round(dist, 1)})
    dist_df = pd.DataFrame(dist_rows)
    st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)

# -------------------------
# === CTA / Footer ===
# -------------------------
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](https://www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis role with MRE Consulting
""")

