import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2  # Haversine
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="Gulf Rig Ops", layout="wide")

st.title("Submarine Sonar to Subsea Sensors — Gulf of Mexico Fleet")
# === TITLE ===
st.title("Submarine Sonar to Subsea Sensors — Gulf of Mexico")
st.markdown("""
**Real-Time Acoustic Energy + Rig Operations Dashboard**  
*From U.S. Navy STS2 sonar processing to MWD drilling optimization*  
Built by a submarine veteran with 14 years in oilfield telemetry
**Real-Time Rig Operations Dashboard**  
*NOAA Buoys + Active Rigs + MWD Telemetry*  
Built by U.S. Navy STS2 | 14 years oilfield telemetry
""")

# === SIDEBAR: FULL ORIGINAL + MWD CONTROLS ===
# === SIDEBAR: 6 CLOSEST BUOYS ===
with st.sidebar:
    st.header("Gulf Buoy Fleet (Select Up to 10)")
    st.header("Select Buoy (6 Closest to Rigs)")
    # 6 buoys closest to real rigs
    buoy_options = {
        "42001 - West Florida Basin": "42001",
        "42001 - Near Neptune TLP": "42001",
        "42002 - Central Gulf": "42002",
        "42003 - East Florida Basin": "42003",
        "42004 - West Florida Shelf": "42004",
        "42005 - Central Gulf Platforms": "42005",
        "42007 - West Florida Basin": "42007",
        "42008 - Central Gulf": "42008",
        "42009 - Eastern Gulf": "42009",
        "42010 - Western Gulf": "42010",
        "42012 - Central Gulf": "42012",
        "42013 - Eastern Gulf": "42013",
        "42019 - West Florida Shelf": "42019",
        "42020 - Central Gulf": "42020",
        "42022 - Eastern Gulf": "42022",
        "42035 - West Florida Shelf": "42035",
        "42039 - Central Gulf": "42039",
        "42039 - Near Thunder Hawk": "42039",
        "42040 - Eastern Gulf": "42040",
        "42045 - Western Gulf": "42045",
        "42046 - Central Gulf": "42046",
        "42047 - Eastern Gulf": "42047"
        "42012 - Central Gulf": "42012",
        "42035 - West Florida Shelf": "42035"
    }
    selected_buoys = st.multiselect(
        "Choose buoys for comparison (Gulf offshore array)",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=10,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
    selected_buoy = st.selectbox(
        "Choose buoy",
        options=list(buoy_options.keys()),
        index=0
    )
    
    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = Real-time signal processing in extreme noise  
    - **MWD Drilling** = Same math: gamma, resistivity, torque  
    - **Energy Tech** = Multi-source fusion for regional ops  
    """)
    st.info("NOAA 420xx series → Gulf Fleet → Analogous to multi-rig sensor arrays")
    buoy_id = buoy_options[selected_buoy]

    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.05, 0.5, 0.1, 0.05)
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)

    if st.button("Refresh All Buoys"):
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.success("Data refreshed!")

# === DATA INGEST: SPECTRAL + REAL-TIME ===
# === DATA INGEST ===
@st.cache_data(ttl=600)
def get_noaa_data(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
@@ -86,20 +64,16 @@
        df = pd.DataFrame(data)
        if df.empty: raise ValueError()
        df["Station"] = station_id
        df["Last Updated"] = datetime.now(ZoneInfo('America/Chicago')).strftime("%Y-%m-%d %H:%M CST/CDT")
        return df
    except:
        st.warning(f"Buoy {station_id} sparse. Using simulated spectrum.")
        freqs = np.linspace(0.03, 0.40, 25)
        energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1)**2) + np.random.normal(0, 0.2, 25)
        return pd.DataFrame({
            "Frequency (Hz)": freqs,
            "Spectral Energy (m²/Hz)": energy.clip(0),
            "Station": [station_id] * 25,
            "Last Updated": [datetime.now(ZoneInfo('America/Chicago')).strftime("%Y-%m-%d %H:%M CST/CDT")] * 25
            "Station": [station_id] * 25
        })

# === REAL-TIME BUOY DATA (ENVIRONMENTAL) ===
@st.cache_data(ttl=600)
def get_realtime_buoy_data(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
@@ -109,56 +83,136 @@
        return {
            "WVHT": latest.get('WVHT', np.nan),
            "DPD": latest.get('DPD', np.nan),
            "APD": latest.get('APD', np.nan),
            "MWD": latest.get('MWD', np.nan),
            "WSPD": latest.get('WSPD', np.nan),
            "GST": latest.get('GST', np.nan),
            "WD": latest.get('WD', np.nan),
            "PRES": latest.get('PRES', np.nan),
            "ATMP": latest.get('ATMP', np.nan),
            "WTMP": latest.get('WTMP', np.nan),
            "PRES": latest.get('PRES', np.nan)
            "MWD": latest.get('MWD', np.nan)
        }
    except:
        return {k: np.nan for k in ["WVHT", "D(pd)", "APD", "MWD", "WSPD", "GST", "WD", "ATMP", "WTMP", "PRES"]}
        return {k: np.nan for k in ["WVHT", "DPD", "WSPD", "GST", "WD", "PRES", "ATMP", "WTMP", "MWD"]}

dfs = [get_noaa_data(buoy) for buoy in selected_buoys]
combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
# Fetch data
spec_df = get_noaa_data(buoy_id)
env_data = get_realtime_buoy_data(buoy_id)

# === LIVE TIMESTAMP ===
st.caption(f"Data last refreshed: {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S CST/CDT')} | NOAA updates hourly")
# === LAYOUT: 3 COLUMNS ===
col_left, col_center, col_right = st.columns([1.8, 2, 1.2])

# === SPECTRAL PLOT ===
if len(selected_buoys) > 1:
    fig1 = px.line(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)", color="Station",
                   title="Multi-Buoy Wave Spectral Energy = Analogous to Multi-Rig MWD Gamma Intensity",
                   labels={"Spectral Energy (m²/Hz)": "Energy (m²/Hz)"},
                   template="plotly_dark")
else:
    fig1 = px.area(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)",
                   title=f"Wave Spectral Energy for Buoy {selected_buoys[0]} = Analogous to MWD Gamma Ray Intensity",
                   labels={"Spectral Energy (m²/Hz)": "Energy (m²/Hz)"},
                   template="plotly_dark")
fig1.update_layout(height=400)
st.plotly_chart(fig1, use_container_width=True)

# === MWD PULSE SIMULATOR ===
st.subheader("Live MWD Mud Pulse Telemetry (6-Pulse Packet)")
# === LEFT: RIG OPS PANEL (REAL NOAA DATA) ===
with col_left:
    st.subheader("Rig Ops — Live Conditions")
    
    # Real NOAA values
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
    current_dir = f"{np.random.randint(0, 360)}°"
    humidity = f"{np.random.randint(60, 95)}%"
    visibility = f"{np.random.uniform(5, 15):.1f} mi"

    # Nearest rig
    buoy_coords = {
        "42001": [25.933, -86.733], "42002": [27.933, -88.233], "42039": [28.18, -88.67],
        "42040": [29.21, -87.55], "42012": [30.05, -87.55], "42035": [29.23, -84.65]
    }
    real_rigs = [
        {"name": "Neptune TLP", "lat": 27.37, "lon": -89.92},
        {"name": "Thunder Hawk SPAR", "lat": 28.18, "lon": -88.67},
        {"name": "King's Quay FPS", "lat": 27.75, "lon": -89.25},
        {"name": "Sailfin FPSO", "lat": 27.80, "lon": -90.20}
    ]
    b_lat, b_lon = buoy_coords[buoy_id]
    nearest_rig = min(real_rigs, key=lambda r: haversine(b_lat, b_lon, r["lat"], r["lon"]))
    dist = haversine(b_lat, b_lon, nearest_rig["lat"], nearest_rig["lon"])

    # Metrics
    st.metric("Wave Height", wave_height, help="Rig motion, BHA run")
    st.metric("Dom. Period", dom_period, help="Wave type")
    st.metric("Wind Speed", wind_speed, help="DP, crane")
    st.metric("Wind Dir", wind_dir, help="Crane swing")
    st.metric("Pressure", pressure, help="Storm forecast")
    st.metric("Wave Dir", wave_dir, help="Vessel heading")
    st.metric("Sea Temp", water_temp, help="Mud, ROV")
    st.metric("Current Speed", current_speed, help="Riser stress")
    st.metric("Air Temp", air_temp, help="Crew comfort")
    st.metric("Humidity", humidity, help="Fog risk")
    st.metric("Visibility", visibility, help="Helicopter ops")
    st.metric("Nearest Rig", nearest_rig["name"], f"{dist:.0f} mi")

    # Drilling Window
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

if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
    bit_pattern = "1010"
    st.warning("Invalid bit pattern. Using '1010'.")
# === CENTER: MAP (ALWAYS ON) + WAVE ENERGY ===
with col_center:
    st.subheader("Map + Wave Energy vs Rig Proximity")

col1, col2 = st.columns([1, 5])
with col1:
    if 'running' not in st.session_state:
        st.session_state.running = True
    if st.button("Stop" if st.session_state.running else "Start"):
        st.session_state.running = not st.session_state.running
    # Map (always visible)
    m = folium.Map(location=[27.0, -88.5], zoom_start=7, tiles="CartoDB dark_matter")
    
    # Buoy
    folium.CircleMarker(
        location=[b_lat, b_lon],
        radius=12,
        popup=f"Buoy {buoy_id}",
        color="cyan",
        fill=True
    ).add_to(m)

frame = st.empty()
status = st.empty()
    # Rigs
    for rig in real_rigs:
        folium.CircleMarker(
            location=[rig["lat"], rig["lon"]],
            radius=14,
            popup=rig["name"],
            color="orange",
            fill=True
        ).add_to(m)

def generate_mwd_packet():
    st_folium(m, width=700, height=350, key="map")

    # Wave Energy vs Rig Proximity
    avg_energy = spec_df["Spectral Energy (m²/Hz)"].mean()
    impact_data = [{
        "Buoy": buoy_id,
        "Avg Wave Energy (m²/Hz)": round(avg_energy, 2),
        "Nearest Rig (mi)": round(dist, 1)
    }]
    impact_df = pd.DataFrame(impact_data)
    fig = px.scatter(impact_df, x="Nearest Rig (mi)", y="Avg Wave Energy (m²/Hz)",
                     hover_data=["Buoy"], title="Wave Energy Impact",
                     template="plotly_dark")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# === RIGHT: MWD (STATIC + ANIMATED) ===
with col_right:
    st.subheader("MWD Mud Pulse Telemetry")

    if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
        bit_pattern = "1010"

    # Static pulse
    t = np.linspace(0, 4, 400)
    signal = np.zeros_like(t)
    for pos in [0.0, 0.5]:
@@ -169,148 +223,47 @@
        mask = (t >= pos) & (t < pos + pulse_width)
        signal[mask] = 0.8 if bit == '1' else -0.8
    signal += np.random.normal(0, noise_level, len(t))
    return pd.DataFrame({"Time (s)": t, "Amplitude": signal})

if st.session_state.running:
    packet = generate_mwd_packet()
    for shift in np.linspace(0, 4, 80):
        if not st.session_state.running: break
        t_shifted = (packet["Time (s)"] - shift) % 4
        visible = (t_shifted >= 0) & (t_shifted <= 2)
        df_plot = pd.DataFrame({"Time (s)": t_shifted[visible], "Amplitude": packet["Amplitude"][visible]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"],
                                 mode='lines', line=dict(color='cyan', width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(x=2, line_dash="dash", line_color="red")
        fig.update_layout(height=300, template="plotly_dark", showlegend=False,
                          xaxis_title="Time (s)", yaxis_title="Pressure", xaxis_range=[0, 2])
        frame.plotly_chart(fig, use_container_width=True)
        if 1.5 < shift < 2.8:
            status.success(f"PACKET DECODED @ {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S')}")
        else:
            status.info("Waiting for packet...")
        time.sleep(0.1)
else:
    packet = generate_mwd_packet()
    df_plot = packet[packet["Time (s)"] <= 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"],
                             mode='lines', line=dict(color='cyan', width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(x=2, line_dash="dash", line_color="red")
    fig.update_layout(height=300, template="plotly_dark", showlegend=False,
                      xaxis_title="Time (s)", yaxis_title="Pressure", xaxis_range=[0, 2])
    frame.plotly_chart(fig, use_container_width=True)
    status.info("Simulator paused.")

# === RIG OPS PANEL: LIVE ENVIRONMENTAL DATA ===
st.subheader("Rig Ops Panel — Live Environmental Conditions")

placeholder = st.empty()

while True:
    with placeholder.container():
        if selected_buoys:
            buoy = selected_buoys[0]
            try:
                data = get_realtime_buoy_data(buoy)
                # Format values
                wave_height = f"{data['WVHT']:.1f} ft" if not pd.isna(data['WVHT']) else "N/A"
                dom_period = f"{data['DPD']:.1f} s" if not pd.isna(data['DPD']) else "N/A"
                wind_speed = f"{data['WSPD']:.1f} kt" if not pd.isna(data['WSPD']) else "N/A"
                wind_gust = f"{data['GST']:.1f} kt" if not pd.isna(data['GST']) else "N/A"
                air_temp = f"{(data['ATMP'] * 9/5 + 32):.1f}°F" if not pd.isna(data['ATMP']) else "N/A"
                water_temp = f"{(data['WTMP'] * 9/5 + 32):.1f}°F" if not pd.isna(data['WTMP']) else "N/A"
                pressure = f"{data['PRES']:.2f} inHg" if not pd.isna(data['PRES']) else "N/A"

                # Nearest rig
                buoy_lat, buoy_lon = buoy_coords[buoy]
                nearest_rig = min(real_rigs, key=lambda r: haversine(buoy_lat, buoy_lon, r["lat"], r["lon"]))
                dist = haversine(buoy_lat, buoy_lon, nearest_rig["lat"], nearest_rig["lon"])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Wave Height", wave_height)
                    st.metric("Dom. Period", dom_period)
                with col2:
                    st.metric("Wind Speed", wind_speed)
                    st.metric("Wind Gust", wind_gust)
                with col3:
                    st.metric("Air Temp", air_temp)
                    st.metric("Water Temp", water_temp)
                with col4:
                    st.metric("Nearest Rig", nearest_rig["name"], f"{dist:.1f} mi")
                    st.metric("Pressure", pressure)

                # Drilling Window
                try:
                    wh = float(wave_height.split()[0]) if wave_height != "N/A" else 99
                    ws = float(wind_speed.split()[0]) if wind_speed != "N/A" else 99
                    if wh < 6.0 and ws < 25:
                        st.success("**DRILLING WINDOW: OPEN** — Safe to drill")
                    elif wh < 8.0 and ws < 30:
                        st.warning("**DRILLING WINDOW: MARGINAL** — Monitor closely")
                    else:
                        st.error("**DRILLING WINDOW: CLOSED** — High risk")
                except:
                    st.info("**DRILLING WINDOW: DATA PENDING**")

            except Exception as e:
                st.error(f"Buoy {buoy} data unavailable. Using simulation.")
                st.metric("Wave Height", "4.2 ft")
                st.metric("Wind Speed", "18 kt")
                st.success("**DRILLING WINDOW: OPEN**")

    time.sleep(600)  # 10-minute refresh

# === MAP + REAL RIGS ===
st.subheader("Buoy & Active Rig Locations (Gulf of Mexico)")
m = folium.Map(location=[25.0, -90.0], zoom_start=5, tiles="CartoDB dark_matter")

buoy_coords = {
    "42001": [25.933, -86.733], "42002": [27.933, -88.233], "42003": [28.933, -84.733],
    "42004": [26.933, -87.733], "42005": [27.933, -88.233], "42007": [25.933, -86.733],
    "42008": [27.933, -88.233], "42009": [28.933, -84.733], "42010": [26.933, -87.733],
    "42012": [27.933, -88.233], "42013": [28.933, -84.733], "42019": [26.933, -87.733],
    "42020": [27.933, -88.233], "42022": [28.933, -84.733], "42035": [26.933, -87.733],
    "42039": [27.933, -88.233], "42040": [28.933, -84.733], "42045": [26.933, -87.733],
    "42046": [27.933, -88.233], "42047": [28.933, -84.733]
}

for buoy in selected_buoys:
    if buoy in buoy_coords:
        folium.CircleMarker(
            location=buoy_coords[buoy],
            radius=10,
            popup=f"NOAA Buoy {buoy}<br>Wave Spectra = Acoustic Proxy",
            color="cyan",
            fill=True
        ).add_to(m)

# === 4 REAL RIGS (BOEM/BSEE 2025) ===
real_rigs = [
    {"name": "Neptune TLP", "operator": "Chevron", "lat": 27.37, "lon": -89.92, "status": "Active", "prod": "10k bbl/day"},
    {"name": "Thunder Hawk SPAR", "operator": "Murphy Oil", "lat": 28.18, "lon": -88.67, "status": "Active", "prod": "8k bbl/day"},
    {"name": "King's Quay FPS", "operator": "Shell", "lat": 27.75, "lon": -89.25, "status": "Active", "prod": "12k bbl/day"},
    {"name": "Sailfin FPSO", "operator": "Beacon Offshore", "lat": 27.80, "lon": -90.20, "status": "Active", "prod": "7k bbl/day"}
]

for rig in real_rigs:
    folium.CircleMarker(
        location=[rig["lat"], rig["lon"]],
        radius=12,
        popup=f"{rig['name']} ({rig['operator']})<br>{rig['status']} | {rig['prod']}<br>Updated: {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M CST/CDT')}",
        color="orange",
        fill=True
    ).add_to(m)

st_folium(m, width=700, height=400)
    static_df = pd.DataFrame({"Time (s)": t[t <= 2], "Amplitude": signal[t <= 2]})
    fig_static = go.Figure()
    fig_static.add_trace(go.Scatter(x=static_df["Time (s)"], y=static_df["Amplitude"],
                                    mode='lines', line=dict(color='cyan', width=3)))
    fig_static.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_static.add_vline(x=0, line_dash="dash", line_color="red")
    fig_static.add_vline(x=2, line_dash="dash", line_color="red")
    fig_static.update_layout(height=200, template="plotly_dark", showlegend=False,
                             xaxis_range=[0, 2], margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_static, use_container_width=True)

    # Animated
    if 'running' not in st.session_state:
        st.session_state.running = True
    if st.button("Stop" if st.session_state.running else "Start", key="mwd_btn"):
        st.session_state.running = not st.session_state.running

# === DISTANCE TABLE + INTEGRATION ===
st.subheader("Buoy-to-Rig Distance Analysis")
    frame = st.empty()
    if st.session_state.running:
        packet = pd.DataFrame({"Time (s)": t, "Amplitude": signal})
        for shift in np.linspace(0, 4, 80):
            if not st.session_state.running: break
            t_shifted = (packet["Time (s)"] - shift) % 4
            visible = (t_shifted >= 0) & (t_shifted <= 2)
            df_plot = pd.DataFrame({"Time (s)": t_shifted[visible], "Amplitude": packet["Amplitude"][visible]})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"],
                                     mode='lines', line=dict(color='cyan', width=3)))
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.add_vline(x=2, line_dash="dash", line_color="red")
            fig.update_layout(height=200, template="plotly_dark", showlegend=False,
                              xaxis_range=[0, 2], margin=dict(l=0, r=0, t=0, b=0))
            frame.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)
    else:
        st.info("Paused")

# === DISTANCE TABLE ===
st.subheader("Buoy-to-Rig Distances")
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
@@ -320,25 +273,16 @@
    return R * c * 0.621371

dist_data = []
for buoy in selected_buoys:
    if buoy in buoy_coords:
        b_lat, b_lon = buoy_coords[buoy]
        for rig in real_rigs:
            dist = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
            dist_data.append({
                "Buoy": buoy,
                "Rig": rig["name"],
                "Distance (mi)": round(dist, 1),
                "Operator": rig["operator"]
            })
for rig in real_rigs:
    dist = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
    dist_data.append({"Rig": rig["name"], "Distance (mi)": round(dist, 1)})

dist_df = pd.DataFrame(dist_data)
if not dist_df.empty:
    st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)
st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)

# === CTA ===
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis position with MRE Consulting
**This is how I processed sonar at 5,000 ft.**  
**Now I'll do it for your rig.**  
[LinkedIn](https://www.linkedin.com/in/nicholas-leiker-50686755) | Seeking MRE Consulting role
""")

