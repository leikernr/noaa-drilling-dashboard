import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

# === PAGE CONFIG ===
st.set_page_config(page_title="Gulf Rig Ops", layout="wide")

# === TITLE ===
st.title("Submarine Sonar to Subsea Sensors — Gulf of Mexico")
st.markdown("""
**Real-Time Rig Operations Dashboard**  
*NOAA Buoys + Active Rigs + MWD Telemetry*  
Built by U.S. Navy STS2 | 14 years oilfield telemetry
""")

# === SIDEBAR: 6 BUOYS + WHY THIS MATTERS (EXACT ORIGINAL) ===
with st.sidebar:
    st.header("Select Buoy (6 Closest to Rigs)")
    buoy_options = {
        "42001 - Near Neptune TLP": "42001",
        "42002 - Central Gulf": "42002",
        "42039 - Near Thunder Hawk": "42039",
        "42040 - Eastern Gulf": "42040",
        "42012 - Central Gulf": "42012",
        "42035 - West Florida Shelf": "42035"
    }
    selected_name = st.selectbox(
        "Choose buoy",
        options=list(buoy_options.keys()),
        index=0
    )
    buoy_id = buoy_options[selected_name]

    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = Real-time signal processing  
    - **MWD Drilling** = Same math: gamma, resistivity, torque  
    - **Energy Tech** = Multi-source fusion for ops  
    """)
    st.info("NOAA 420xx → Gulf Fleet → Multi-rig sensor analogy")

    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.05, 0.5, 0.1, 0.05)
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.success("Data refreshed!")

# === COORDINATES & RIGS (ALL 6 BUOYS + 4 RIGS) ===
buoy_coords = {
    "42001": [25.933, -86.733],
    "42002": [26.055, -90.333],
    "42039": [28.790, -86.007],
    "42040": [29.212, -88.208],
    "42012": [30.059, -87.548],
    "42035": [29.232, -84.650]
}

real_rigs = [
    {"name": "Neptune TLP", "lat": 27.37, "lon": -89.92},
    {"name": "Thunder Hawk SPAR", "lat": 28.18, "lon": -88.67},
    {"name": "King's Quay FPS", "lat": 27.75, "lon": -89.25},
    {"name": "Sailfin FPSO", "lat": 27.80, "lon": -90.20}
]

# === DATA INGEST: SPECTRAL + REALTIME ===
@st.cache_data(ttl=600)
def get_noaa_data(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
    try:
        response = requests.get(url, timeout=10)
        lines = response.text.splitlines()
        data = []
        for line in lines[2:]:
            if line.strip() and not line.startswith('#'):
                cols = line.split()
                if len(cols) >= 2:
                    data.append({"Frequency (Hz)": float(cols[0]), "Spectral Energy (m²/Hz)": float(cols[1])})
        df = pd.DataFrame(data)
        if df.empty: raise ValueError()
        df["Station"] = station_id
        return df
    except:
        freqs = np.linspace(0.03, 0.40, 25)
        energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1)**2) + np.random.normal(0, 0.2, 25)
        return pd.DataFrame({
            "Frequency (Hz)": freqs,
            "Spectral Energy (m²/Hz)": energy.clip(0),
            "Station": [station_id] * 25
        })

@st.cache_data(ttl=600)
def get_realtime_buoy_data(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    try:
        df = pd.read_csv(url, delim_whitespace=True, skiprows=[1], na_values=['MM'])
        latest = df.iloc[-1]
        return {
            "WVHT": latest.get('WVHT', np.nan),
            "DPD": latest.get('DPD', np.nan),
            "WSPD": latest.get('WSPD', np.nan),
            "GST": latest.get('GST', np.nan),
            "WD": latest.get('WD', np.nan),
            "PRES": latest.get('PRES', np.nan),
            "ATMP": latest.get('ATMP', np.nan),
            "WTMP": latest.get('WTMP', np.nan),
            "MWD": latest.get('MWD', np.nan)
        }
    except:
        return {k: np.nan for k in ["WVHT", "DPD", "WSPD", "GST", "WD", "PRES", "ATMP", "WTMP", "MWD"]}

# Fetch data
spec_df = get_noaa_data(buoy_id)
env_data = get_realtime_buoy_data(buoy_id)

# === LAYOUT: 3 COLUMNS ===
col_left, col_center, col_right = st.columns([1.8, 2, 1.2])

# === LEFT: RIG OPS PANEL (FULL 11 METRICS) ===
with col_left:
    st.subheader("Rig Ops — Live Environmental Conditions")

    wave_height = f"{env_data['WVHT']:.1f} ft" if not pd.isna(env_data['WVHT']) else "—"
    dom_period = f"{env_data['DPD']:.1f} s" if not pd.isna(env_data['DPD']) else "—"
    wind_speed = f"{env_data['WSPD']:.1f} kt" if not pd.isna(env_data['WSPD']) else "—"
    wind_dir = f"{int(env_data['WD'])}°" if not pd.isna(env_data['WD']) else "—"
    pressure = f"{env_data['PRES']:.2f} inHg" if not pd.isna(env_data['PRES']) else "—"
    wave_dir = f"{int(env_data['MWD'])}°" if not pd.isna(env_data['MWD']) else "—"
    water_temp = f"{(env_data['WTMP'] * 9/5 + 32):.1f}°F" if not pd.isna(env_data['WTMP']) else "—"
    air_temp = f"{(env_data['ATMP'] * 9/5 + 32):.1f}°F" if not pd.isna(env_data['ATMP']) else "—"

    # Simulated
    current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
    humidity = f"{np.random.randint(60, 95)}%"
    visibility = f"{np.random.uniform(5, 15):.1f} mi"

    # Nearest rig
    b_lat, b_lon = buoy_coords[buoy_id]
    nearest_rig = min(real_rigs, key=lambda r: haversine(b_lat, b_lon, r["lat"], r["lon"]))
    dist = haversine(b_lat, b_lon, nearest_rig["lat"], nearest_rig["lon"])

    # === FULL METRICS ===
    st.metric("Wave Height", wave_height, help="Rig motion, BHA run")
    st.metric("Dom. Period", dom_period, help="Wave type (swell/chop)")
    st.metric("Wind Speed", wind_speed, help="DP systems, crane ops")
    st.metric("Wind Dir", wind_dir, help="Crane swing direction")
    st.metric("Barometric Pressure", pressure, help="Storm forecasting")
    st.metric("Wave Direction", wave_dir, help="Vessel heading, stability")
    st.metric("Sea Surface Temp", water_temp, help="Weather, mud cooling")
    st.metric("Current Speed", current_speed, help="Riser stress, DP")
    st.metric("Air Temp", air_temp, help="Comfort, fog risk")
    st.metric("Humidity", humidity, help="Fog risk, comfort")
    st.metric("Visibility", visibility, help="Helicopter ops")
    st.metric("Nearest Rig", nearest_rig["name"], f"{dist:.0f} mi")

    # === DRILLING WINDOW ===
    try:
        wh = float(wave_height.split()[0]) if wave_height != "—" else 99
        ws = float(wind_speed.split()[0]) if wind_speed != "—" else 99
        if wh < 6.0 and ws < 25:
            st.success("**DRILLING WINDOW: OPEN** — Safe to drill")
        elif wh < 8.0 and ws < 30:
            st.warning("**DRILLING WINDOW: MARGINAL** — Monitor closely")
        else:
            st.error("**DRILLING WINDOW: CLOSED** — High risk")
    except:
        st.info("**DRILLING WINDOW: DATA PENDING**")

# === CENTER: MAP (ALWAYS ON) + WAVE ENERGY + RESISTIVITY PULSE ===
with col_center:
    st.subheader("Map + Wave Energy + Resistivity Pulse")

    # Map
    m = folium.Map(location=[27.5, -88.5], zoom_start=7, tiles="CartoDB dark_matter")
    folium.CircleMarker(
        location=[b_lat, b_lon],
        radius=12,
        popup=f"Buoy {buoy_id}<br>Wave Spectra = Acoustic Proxy",
        color="cyan",
        fill=True
    ).add_to(m)
    for rig in real_rigs:
        folium.CircleMarker(
            location=[rig["lat"], rig["lon"]],
            radius=14,
            popup=f"{rig['name']}<br>Updated: {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M CST/CDT')}",
            color="orange",
            fill=True
        ).add_to(m)
    st_folium(m, width=700, height=350, key="map")

    # Wave Energy vs Rig Proximity
    avg_energy = spec_df["Spectral Energy (m²/Hz)"].mean()
    impact_data = [{
        "Buoy": buoy_id,
        "Avg Wave Energy (m²/Hz)": round(avg_energy, 2),
        "Nearest Rig (mi)": round(dist, 1)
    }]
    impact_df = pd.DataFrame(impact_data)
    fig_wave = px.scatter(impact_df, x="Nearest Rig (mi)", y="Avg Wave Energy (m²/Hz)",
                          hover_data=["Buoy"], title="Wave Energy vs Rig Proximity",
                          template="plotly_dark")
    fig_wave.update_layout(height=300)
    st.plotly_chart(fig_wave, use_container_width=True)

    # === RESISTIVITY PULSE (6-PACK PULSE) ===
    st.markdown("**Resistivity Tool Pulse (6-Pack)**")
    t_res = np.linspace(0, 3, 300)
    ping = np.zeros_like(t_res)
    for i in range(6):
        pos = 0.5 * i
        mask = (t_res >= pos) & (t_res < pos + 0.3)
        ping[mask] = 1.0
    ping += np.random.normal(0, 0.1, len(t_res))
    ping_df = pd.DataFrame({"Time (s)": t_res, "Amplitude": ping})

    fig_resistivity = go.Figure()
    fig_resistivity.add_trace(go.Scatter(x=ping_df["Time (s)"], y=ping_df["Amplitude"],
                                         mode='lines', name='Resistivity Pulse', line=dict(color='cyan')))
    fig_resistivity.update_layout(title="Like Sending a Resistivity Tool Pulse Downhole", height=300, template="plotly_dark")
    st.plotly_chart(fig_resistivity, use_container_width=True)

# === RIGHT: MWD — STATIC (ALWAYS ON) + ANIMATED (ON BUTTON) ===
with col_right:
    st.subheader("MWD Mud Pulse Telemetry")

    # Validate bit pattern
    if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
        bit_pattern = "1010"

    # Generate full packet
    t = np.linspace(0, 4, 400)
    signal = np.zeros_like(t)
    for pos in [0.0, 0.5]:
        mask = (t >= pos) & (t < pos + pulse_width * 1.5)
        signal[mask] = 1.0
    for i, bit in enumerate(bit_pattern):
        pos = 1.0 + i * 0.5
        mask = (t >= pos) & (t < pos + pulse_width)
        signal[mask] = 0.8 if bit == '1' else -0.8
    signal += np.random.normal(0, noise_level, len(t))

    # === 1. STATIC PULSE (ALWAYS VISIBLE) ===
    st.markdown("**Static MWD Packet (Reference)**")
    static_df = pd.DataFrame({"Time (s)": t[t <= 2], "Amplitude": signal[t <= 2]})
    fig_static = go.Figure()
    fig_static.add_trace(go.Scatter(x=static_df["Time (s)"], y=static_df["Amplitude"],
                                    mode='lines', line=dict(color='cyan', width=3)))
    fig_static.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_static.add_vline(x=0, line_dash="dash", line_color="red")
    fig_static.add_vline(x=2, line_dash="dash", line_color="red")
    fig_static.update_layout(height=180, template="plotly_dark", showlegend=False,
                             xaxis_range=[0, 2], margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_static, use_container_width=True)

    # === 2. ANIMATED PULSE (START/STOP BUTTON) ===
    st.markdown("**Live Telemetry Stream**")
    if 'animating' not in st.session_state:
        st.session_state.animating = False
    if 'frame' not in st.session_state:
        st.session_state.frame = 0

    col_btn = st.columns([1, 1])
    with col_btn[0]:
        if st.button("Start" if not st.session_state.animating else "Stop", key="anim_btn"):
            st.session_state.animating = not st.session_state.animating
            if not st.session_state.animating:
                st.session_state.frame = 0

    anim_placeholder = st.empty()

    if st.session_state.animating:
        shift = (st.session_state.frame % 80) * (4 / 80)
        t_shifted = (t - shift) % 4
        visible = (t_shifted >= 0) & (t_shifted <= 2)
        df_plot = pd.DataFrame({"Time (s)": t_shifted[visible], "Amplitude": signal[visible]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"],
                                 mode='lines', line=dict(color='cyan', width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(x=2, line_dash="dash", line_color="red")
        fig.update_layout(height=180, template="plotly_dark", showlegend=False,
                          xaxis_range=[0, 2], margin=dict(l=0, r=0, t=0, b=0))
        anim_placeholder.plotly_chart(fig, use_container_width=True)

        st.session_state.frame += 1
        st.rerun()
    else:
        st.info("Press 'Start' to begin telemetry stream")

# === DISTANCE TABLE (FULL) ===
st.subheader("Buoy-to-Rig Distance Analysis")
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c * 0.621371

dist_data = []
for rig in real_rigs:
    dist = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
    dist_data.append({"Rig": rig["name"], "Distance (mi)": round(dist, 1)})

dist_df = pd.DataFrame(dist_data)
st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)

# === CTA ===
st.success("""
**This is how I processed sonar at 5,000 ft.**  
**Now I'll do it for your rig.**  
[LinkedIn](https://www.linkedin.com/in/nicholas-leiker-50686755) | Seeking MRE Consulting role
""")

