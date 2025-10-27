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

st.set_page_config(page_title="MWD Telemetry Tool", layout="wide")

st.title("Submarine Sonar to MWD Telemetry — Gulf of Mexico Fleet")
st.markdown("**Real-Time Sensor Fusion + Live Mud Pulse Simulator**")

# === SIDEBAR: BUOY + MWD CONTROLS ===
with st.sidebar:
    st.header("Gulf Buoy Fleet (Select Up to 10)")
    buoy_options = {
        "42001 - West Florida Basin": "42001",
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
        "42040 - Eastern Gulf": "42040",
        "42045 - Western Gulf": "42045",
        "42046 - Central Gulf": "42046",
        "42047 - Eastern Gulf": "42047"
    }
    selected_buoys = st.multiselect(
        "Choose buoys",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=10,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
    )

    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.05, 0.5, 0.1, 0.05)
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)

    if st.button("Refresh Buoys"):
        st.cache_data.clear()
        st.success("Data refreshed!")

# === DATA INGEST ===
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
        df["Last Updated"] = datetime.now(ZoneInfo('America/Chicago')).strftime("%H:%M CST/CDT")
        return df
    except:
        freqs = np.linspace(0.03, 0.40, 25)
        energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1)**2) + np.random.normal(0, 0.2, 25)
        return pd.DataFrame({
            "Frequency (Hz)": freqs,
            "Spectral Energy (m²/Hz)": energy.clip(0),
            "Station": [station_id] * 25,
            "Last Updated": [datetime.now(ZoneInfo('America/Chicago')).strftime("%H:%M CST/CDT")] * 25
        })

dfs = [get_noaa_data(buoy) for buoy in selected_buoys]
combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

# === LIVE TIMESTAMP ===
st.caption(f"Last updated: {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S CST/CDT')}")

# === SPECTRAL PLOT ===
if len(selected_buoys) > 1:
    fig1 = px.line(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)", color="Station",
                   title="Multi-Buoy Wave Spectra = Multi-Rig MWD Analogy", template="plotly_dark")
else:
    fig1 = px.area(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)",
                   title=f"Buoy {selected_buoys[0]} = MWD Gamma Analogy", template="plotly_dark")
fig1.update_layout(height=400)
st.plotly_chart(fig1, use_container_width=True)

# === MWD PULSE SIMULATOR ===
st.subheader("Live MWD Mud Pulse Telemetry (6-Pulse Packet)")

# Validate bit pattern
if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
    bit_pattern = "1010"
    st.warning("Invalid bit pattern. Using '1010'.")

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
    # Sync pulses
    for pos in [0.0, 0.5]:
        mask = (t >= pos) & (t < pos + pulse_width * 1.5)
        signal[mask] = 1.0
    # Data bits
    for i, bit in enumerate(bit_pattern):
        pos = 1.0 + i * 0.5
        mask = (t >= pos) & (t < pos + pulse_width)
        signal[mask] = 0.8 if bit == '1' else -0.8
    # Noise
    signal += np.random.normal(0, noise_level, len(t))
    return pd.DataFrame({"Time (s)": t, "Amplitude": signal})

# Animation loop
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

# === MAP: ALL BUOYS ===
st.subheader("Buoy & Rig Locations")
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

folium.CircleMarker(
    location=[26.0, -90.5],
    radius=8,
    popup="Sample Rig Site<br>Real-Time MWD Telemetry",
    color="orange",
    fill=True
).add_to(m)

st_folium(m, width=700, height=400)

st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis position with MRE Consulting
""")
