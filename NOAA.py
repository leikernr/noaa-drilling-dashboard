import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

# ========================================
# PAGE CONFIG & TITLE
# ========================================
st.set_page_config(page_title="NOAA RigOps Dashboard", layout="wide")
st.title("NOAA Offshore Drilling Dashboard")
st.caption("Real-time marine data fused with MWD telemetry simulation")

# ========================================
# RIGS & BUOYS (6 CLOSEST)
# ========================================
real_rigs = [
    {"name": "Olympus TLP", "lat": 27.22, "lon": -90.00},
    {"name": "Mars TLP", "lat": 27.18, "lon": -89.25},
    {"name": "Ursa", "lat": 27.33, "lon": -89.21},
    {"name": "Appomattox", "lat": 27.00, "lon": -88.34},
]

buoy_info = {
    "42001": ("42001 - Near Olympus TLP", 25.933, -86.733),
    "42002": ("42002 - Central Gulf", 26.055, -90.333),
    "42039": ("42039 - Near Thunder Hawk", 28.790, -86.007),
    "42040": ("42040 - Eastern Gulf", 29.212, -88.208),
    "42012": ("42012 - Central Gulf", 30.059, -87.548),
    "42035": ("42035 - West Florida Shelf", 29.232, -84.650)
}

# Haversine distance (miles)
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ========================================
# SIDEBAR: CONTROLS
# ========================================
with st.sidebar:
    st.header("Controls")

    # Buoy selector with named locations
    buoy_options = {info[0]: bid for bid, info in buoy_info.items()}
    selected_buoys = st.multiselect(
        "Choose buoys (6 closest)",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=6,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
    )

    # Why This Matters
    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = Real-time signal processing  
    - **MWD Drilling** = Same math: gamma, resistivity, torque  
    - **Energy Tech** = Multi-source fusion for ops  
    """)
    st.info("NOAA 420xx → Gulf Fleet → Multi-rig sensor analogy")

    # MWD Simulator Controls
    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.05, 0.5, 0.10, 0.05)
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)

    if st.button("Refresh All"):
        st.cache_data.clear()
        st.rerun()

if not selected_buoys:
    selected_buoys = ["42001"]

# ========================================
# NOAA DATA FETCH (10–20 MIN CACHE)
# ========================================
@st.cache_data(ttl=600)
def fetch_spectral(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        lines = r.text.splitlines()
        data = []
        for line in lines[2:]:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    data.append({"Frequency (Hz)": float(parts[0]), "Spectral Energy (m²/Hz)": float(parts[1])})
        df = pd.DataFrame(data)
        if not df.empty:
            df["Station"] = station_id
            return df
    except:
        pass
    freqs = np.linspace(0.03, 0.40, 25)
    energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1)**2) + np.random.normal(0, 0.2, 25)
    df = pd.DataFrame({"Frequency (Hz)": freqs, "Spectral Energy (m²/Hz)": energy.clip(0)})
    df["Station"] = station_id
    return df

@st.cache_data(ttl=600)
def fetch_realtime(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    try:
        df = pd.read_csv(url, delim_whitespace=True, skiprows=[1], na_values=["MM"], comment="#")
        if df.empty: raise ValueError()
        latest = df.iloc[-1]
        return {
            "WVHT": latest.get("WVHT", np.nan),
            "DPD": latest.get("DPD", np.nan),
            "WSPD": latest.get("WSPD", np.nan),
            "WD": latest.get("WD", np.nan),
            "PRES": latest.get("PRES", np.nan),
            "ATMP": latest.get("ATMP", np.nan),
            "WTMP": latest.get("WTMP", np.nan),
            "MWD": latest.get("MWD", np.nan)
        }
    except:
        return {k: np.nan for k in ["WVHT", "DPD", "WSPD", "WD", "PRES", "ATMP", "WTMP", "MWD"]}

# ========================================
# LAYOUT: 3 COLUMNS
# ========================================
col_left, col_center, col_right = st.columns([1.8, 2.5, 1.2])

# ========================================
# LEFT: NOAA BUOY DATA (STABLE)
# ========================================
with col_left:
    st.subheader("NOAA Buoy Data — Live Environmental Conditions")
    primary = selected_buoys[0]
    rt = fetch_realtime(primary)
    b_lat, b_lon = buoy_info[primary][1], buoy_info[primary][2]

    # Format NOAA values
    wave_height = f"{rt['WVHT']:.1f} ft" if not pd.isna(rt['WVHT']) else "—"
    dom_period = f"{rt['DPD']:.1f} s" if not pd.isna(rt['DPD']) else "—"
    wind_speed = f"{rt['WSPD']:.1f} kt" if not pd.isna(rt['WSPD']) else "—"
    wind_dir = f"{int(rt['WD'])}°" if not pd.isna(rt['WD']) else "—"
    pressure = f"{rt['PRES']:.2f} inHg" if not pd.isna(rt['PRES']) else "—"
    wave_dir = f"{int(rt['MWD'])}°" if not pd.isna(rt['MWD']) else "—"
    sea_temp = f"{(rt['WTMP'] * 9/5 + 32):.1f}°F" if not pd.isna(rt['WTMP']) else "—"
    air_temp = f"{(rt['ATMP'] * 9/5 + 32):.1f}°F" if not pd.isna(rt['ATMP']) else "—"

    # Simulated extras
    current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
    humidity = f"{np.random.randint(60, 95)}%"
    visibility = f"{np.random.uniform(5, 15):.1f} mi"

    # Nearest rig
    nearest_rig = min(real_rigs, key=lambda r: haversine(b_lat, b_lon, r["lat"], r["lon"]))
    dist = haversine(b_lat, b_lon, nearest_rig["lat"], nearest_rig["lon"])

    # 3-column metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Wave Height", wave_height)
        st.metric("Dom. Period", dom_period)
        st.metric("Wind Speed", wind_speed)
        st.metric("Wind Dir", wind_dir)
    with col2:
        st.metric("Barometric Pressure", pressure)
        st.metric("Wave Dir", wave_dir)
        st.metric("Sea Temp", sea_temp)
        st.metric("Current Speed", current_speed)
    with col3:
        st.metric("Air Temp", air_temp)
        st.metric("Humidity", humidity)
        st.metric("Visibility", visibility)
        st.metric("Nearest Rig", f"{nearest_rig['name']} ({dist:.0f} mi)")

    # Drilling window
    try:
        wh = float(wave_height.split()[0]) if wave_height != "—" else 99
        ws = float(wind_speed.split()[0]) if wind_speed != "—" else 99
        if wh < 6.0 and ws < 25:
            st.success("**DRILLING WINDOW: OPEN**")
        elif wh < 8.0 and ws < 30:
            st.warning("**DRILLING WINDOW: MARGINAL**")
        else:
            st.error("**DRILLING WINDOW: CLOSED**")
    except:
        st.info("**DRILLING WINDOW: PENDING**")

# ========================================
# CENTER: SIMULATED PINGS + ANALYSIS + MAP
# ========================================
with col_center:
    st.subheader("Simulated Active Sonar Ping (MWD Pulse)")

    # 1. MWD PULSE (SINE DECAY) — CORRECTED TITLE
    t = np.linspace(0, 2, 200)
    ping = np.sin(2 * np.pi * 5 * t) * np.exp(-t*3)
    ping_df = pd.DataFrame({"Time (s)": t, "Amplitude": ping})
    fig_mwd = go.Figure()
    fig_mwd.add_trace(go.Scatter(x=ping_df["Time (s)"], y=ping_df["Amplitude"],
                                 mode='lines', name='MWD Pulse', line=dict(color='cyan')))
    fig_mwd.update_layout(
        title="MWD Pulse Downhole (Analog to Resistivity)",
        height=300,
        template="plotly_dark"
    )
    st.plotly_chart(fig_mwd, use_container_width=True)

    # 2. 6-PACK MWD PULSE
    st.markdown("**MWD 6-Pack Telemetry**")
    t_res = np.linspace(0, 3, 300)
    ping6 = np.zeros_like(t_res)
    for i in range(6):
        pos = 0.5 * i
        mask = (t_res >= pos) & (t_res < pos + 0.3)
        ping6[mask] = 1.0
    ping6 += np.random.normal(0, 0.1, len(t_res))
    ping6_df = pd.DataFrame({"Time (s)": t_res, "Amplitude": ping6})
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=ping6_df["Time (s)"], y=ping6_df["Amplitude"],
                              mode='lines', name='6-Pack', line=dict(color='cyan')))
    fig6.update_layout(height=300, template="plotly_dark")
    st.plotly_chart(fig6, use_container_width=True)

    # 3. WAVE ENERGY vs RIG PROXIMITY
    spectral_dfs = [fetch_spectral(b) for b in selected_buoys]
    combined_df = pd.concat(spectral_dfs) if spectral_dfs else pd.DataFrame()
    impact_rows = []
    for bid in selected_buoys:
        energy = combined_df[combined_df["Station"] == bid]["Spectral Energy (m²/Hz)"].mean()
        b_lat, b_lon = buoy_info[bid][1], buoy_info[bid][2]
        dist = min([haversine(b_lat, b_lon, r["lat"], r["lon"]) for r in real_rigs])
        impact_rows.append({"Buoy": bid, "Avg Energy": round(energy, 2) if not pd.isna(energy) else np.nan, "Nearest Rig (mi)": round(dist, 1)})
    impact_df = pd.DataFrame(impact_rows).dropna()
    if not impact_df.empty:
        fig_wave = px.scatter(impact_df, x="Nearest Rig (mi)", y="Avg Energy", hover_data=["Buoy"],
                              title="Wave Energy vs Rig Proximity", template="plotly_dark")
        fig_wave.update_layout(height=400)
        st.plotly_chart(fig_wave, use_container_width=True)

    # 4. BUOY-TO-RIG DISTANCE TABLE
    st.subheader("Buoy-to-Rig Distances")
    dist_rows = []
    for bid in selected_buoys:
        b_lat, b_lon = buoy_info[bid][1], buoy_info[bid][2]
        for rig in real_rigs:
            d = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
            dist_rows.append({"Buoy": bid, "Rig": rig["name"], "Distance (mi)": round(d, 1)})
    dist_df = pd.DataFrame(dist_rows)
    st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True, height=400)

    # 5. LARGE MAP — ALWAYS ON
    st.subheader("Gulf of Mexico — Rigs & Buoys")
    m = folium.Map(location=[27.5, -88.5], zoom_start=7, tiles="CartoDB dark_matter")
    for rig in real_rigs:
        folium.CircleMarker([rig["lat"], rig["lon"]], radius=12, popup=rig["name"], color="orange", fill=True).add_to(m)
    for bid in selected_buoys:
        lat, lon = buoy_info[bid][1], buoy_info[bid][2]
        folium.CircleMarker([lat, lon], radius=10, popup=f"Buoy {bid}", color="cyan", fill=True).add_to(m)
    st_folium(m, width=1000, height=500, key="map")

# ========================================
# RIGHT: MWD PULSE SIMULATOR (FIXED)
# ========================================
with col_right:
    st.subheader("MWD Mud Pulse Telemetry Simulator")

    if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
        bit_pattern = "1010"

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

    # Static pulse
    st.markdown("**Static MWD Packet**")
    static_df = pd.DataFrame({"Time (s)": t[t <= 2], "Amplitude": signal[t <= 2]})
    fig_static = go.Figure()
    fig_static.add_trace(go.Scatter(x=static_df["Time (s)"], y=static_df["Amplitude"], mode='lines', line=dict(color='cyan', width=3)))
    fig_static.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_static.add_vline(x=0, line_dash="dash", line_color="red")
    fig_static.add_vline(x=2, line_dash="dash", line_color="red")
    fig_static.update_layout(height=180, template="plotly_dark", showlegend=False, xaxis_range=[0, 2])
    st.plotly_chart(fig_static, use_container_width=True)

    # Animated pulse
    st.markdown("**Live Telemetry Stream**")
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame' not in st.session_state:
        st.session_state.frame = 0

    if st.button("Start" if not st.session_state.running else "Stop"):
        st.session_state.running = not st.session_state.running
        if not st.session_state.running:
            st.session_state.frame = 0

    anim_placeholder = st.empty()

    if st.session_state.running:
        shift = (st.session_state.frame % 80) * (4 / 80)
        t_shifted = (t - shift) % 4
        visible = (t_shifted >= 0) & (t_shifted <= 2)
        df_plot = pd.DataFrame({"Time (s)": t_shifted[visible], "Amplitude": signal[visible]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"], mode='lines', line=dict(color='cyan', width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(x=2, line_dash="dash", line_color="red")
        fig.update_layout(height=180, template="plotly_dark", showlegend=False, xaxis_range=[0, 2])
        anim_placeholder.plotly_chart(fig, use_container_width=True)
        st.session_state.frame += 1
    else:
        st.info("Press 'Start' to begin stream")

# ========================================
# FOOTER / CTA
# ========================================
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](https://www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis role with MRE Consulting
""")
