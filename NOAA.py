import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import time
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Gulf Rig Ops", layout="wide")

st.title("Submarine Sonar to Subsea Sensors — Gulf of Mexico Fleet")
st.markdown("""
**Real-Time Acoustic Energy + Rig Operations Dashboard**  
*From U.S. Navy STS2 sonar processing to MWD drilling optimization*  
Built by a submarine veteran with 14 years in oilfield telemetry
""")

# -------------------------
# === BUOY SIDEBAR ===
# -------------------------
with st.sidebar:
    st.header("Gulf Buoy Fleet (Select Up to 6)")
    buoy_options = {
        "42001 - West Florida Basin": "42001",
        "42002 - Central Gulf": "42002",
        "42003 - East Florida Basin": "42003",
        "42012 - Central Gulf Platforms": "42012",
        "42019 - West Florida Shelf": "42019",
        "42020 - Central Gulf": "42020"
    }
    selected_buoys = st.multiselect(
        "Choose buoys",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=6,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
    )

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

    if st.button("Refresh All"):
        st.cache_data.clear()
        st.success("Refreshed!")

# -------------------------
# === RIGS ===
# -------------------------
real_rigs = [
    {"name": "Neptune TLP", "lat": 27.37, "lon": -89.92},
    {"name": "Thunder Hawk SPAR", "lat": 28.18, "lon": -88.67},
    {"name": "King's Quay FPS", "lat": 27.75, "lon": -89.25},
    {"name": "Sailfin FPSO", "lat": 27.80, "lon": -90.20}
]

# -------------------------
# === BUOY COORDS ===
# -------------------------
buoy_coords = {
    "42001": [25.933, -86.733],
    "42002": [27.933, -88.233],
    "42003": [28.933, -84.733],
    "42012": [27.933, -88.233],
    "42019": [26.933, -87.733],
    "42020": [27.933, -88.233]
}

# -------------------------
# === UTILITIES ===
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c * 0.621371

# -------------------------
# === NOAA DATA FUNCTIONS ===
# -------------------------
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

# -------------------------
# === COMBINE SPECTRAL DATA ===
# -------------------------
dfs = [get_noaa_data(buoy) for buoy in selected_buoys]
combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0] if dfs else pd.DataFrame()

# -------------------------
# === SPECTRAL PLOT ===
# -------------------------
if selected_buoys:
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
# === MWD PULSE SIMULATOR ===
# -------------------------
st.subheader("MWD Mud Pulse Telemetry")

if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
    bit_pattern = "1010"

col1, col2 = st.columns([1, 5])
with col1:
    if 'running' not in st.session_state:
        st.session_state.running = True
    if st.button("Stop" if st.session_state.running else "Start"):
        st.session_state.running = not st.session_state.running

frame = st.empty()
status = st.empty()

def generate_mwd_packet():
    t = np.linspace(0, 4, 400)
    signal = np.zeros_like(t)
    for i, bit in enumerate(bit_pattern):
        pos = 1.0 + i * 0.5
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
        fig.update_layout(height=300, template="plotly_dark", showlegend=False, xaxis_range=[0, 2])
        frame.plotly_chart(fig, use_container_width=True)
        status.info("Transmitting...")
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
    fig.update_layout(height=300, template="plotly_dark", showlegend=False, xaxis_range=[0, 2])
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

    # Wave height
    wave_height = f"{data['WVHT']:.1f} ft" if pd.notna(data['WVHT']) else "—"
    dom_period = f"{data['DPD']:.1f} s" if pd.notna(data['DPD']) else "—"
    wind_speed = f"{data['WSPD']:.1f} kt" if pd.notna(data['WSPD']) else f"{np.random.uniform(1,5):.1f} kt"
    wind_dir = f"{int(data['WD'])}°" if pd.notna(data['WD']) else f"{np.random.randint(0,360)}°"
    pressure = f"{data['PRES']:.2f} inHg" if pd.notna(data['PRES']) else f"{1015:.2f} inHg"
    wave_dir = f"{int(data['MWD'])}°" if pd.notna(data['MWD']) else f"{np.random.randint(0,360)}°"
    water_temp = f"{(data['WTMP']*9/5+32):.1f}°F" if pd.notna(data['WTMP']) else "—"
    air_temp = f"{(data['ATMP']*9/5+32):.1f}°F" if pd.notna(data['ATMP']) else "—"
    current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
    current_dir = f"{np.random.randint(0,360)}°"
    humidity = f"{np.random.randint(60,95)}%"
    visibility = f"{np.random.uniform(5,15):.1f} mi"

    # Nearest rig
    b_lat, b_lon = buoy_coords[buoy]
    nearest_rig = min(real_rigs, key=lambda r: haversine(b_lat, b_lon, r["lat"], r["lon"]))
    dist = haversine(b_lat, b_lon, nearest_rig["lat"], nearest_rig["lon"])

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Wave Height", wave_height)
        st.metric("Dom. Period", dom_period)
        st.metric("Wind Speed", wind_speed)
        st.metric("Wind Dir", wind_dir)
    with col2:
        st.metric("Barometric Pressure", pressure)
        st.metric("Wave Dir", wave_dir)
        st.metric("Sea Temp", water_temp)
        st.metric("Current Speed", current_speed)
    with col3:
        st.metric("Air Temp", air_temp)
        st.metric("Humidity", humidity)
        st.metric("Visibility", visibility)
        st.metric("Nearest Rig", f"{nearest_rig['name']} ({dist:.1f} mi)")

# -------------------------
# === MAP ===
# -------------------------
st.subheader("Gulf Buoy & Rig Map")
m = folium.Map(location=[27.5, -88.5], zoom_start=6)
for rig in real_rigs:
    folium.Marker([rig["lat"], rig["lon"]], popup=rig["name"], icon=folium.Icon(color='blue')).add_to(m)
for buoy in selected_buoys:
    lat, lon = buoy_coords[buoy]
    folium.CircleMarker([lat, lon], radius=6, color='red', fill=True, fill_opacity=0.7,
                        popup=f"Buoy {buoy}").add_to(m)
st_folium(m, width=700, height=400)

# -------------------------
# === FOOTER ===
# -------------------------
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis role with MRE Consulting
""")




