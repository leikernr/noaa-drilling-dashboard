import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo  # Houston time zone
import time  # For animation
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Sonar to Sensors Multi-Buoy", layout="wide")

st.title("Submarine Sonar to Subsea Sensors — Gulf of Mexico Fleet")
st.markdown("""
**Real-Time Acoustic Energy Dashboard**  
*From U.S. Navy STS2 sonar processing to MWD drilling optimization*  
Built by a submarine veteran with 14 years in oilfield telemetry
""")

# === SIDEBAR: BUOY SELECTOR ===
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
        "Choose buoys for comparison (Gulf offshore array)",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=10,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
    )
    
    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = Real-time signal processing in extreme noise  
    - **MWD Drilling** = Same math: gamma, resistivity, torque  
    - **Energy Tech** = Multi-source fusion for regional ops  
    """)
    st.info("NOAA 420xx series → Gulf Fleet → Analogous to multi-rig sensor arrays")

# === MANUAL REFRESH BUTTON ===
if st.button("Refresh All Buoys"):
    st.cache_data.clear()
    st.success("Data refreshed!")

# === DATA INGEST (Multi-Buoy) ===
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
                    freq = float(cols[0])
                    energy = float(cols[1])
                    data.append({"Frequency (Hz)": freq, "Spectral Energy (m²/Hz)": energy})
        df = pd.DataFrame(data)
        if df.empty or len(df) < 3:
            raise ValueError("Sparse data")
        df["Station"] = station_id
        df["Last Updated"] = datetime.now(ZoneInfo('America/Chicago')).strftime("%Y-%m-%d %H:%M CST/CDT")
        return df
    except:
        st.warning(f"Buoy {station_id} sparse. Using simulated spectrum.")
        freqs = np.linspace(0.03, 0.40, 25)
        energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1)**2) + np.random.normal(0, 0.2, 25)
        df = pd.DataFrame({
            "Frequency (Hz)": freqs,
            "Spectral Energy (m²/Hz)": energy.clip(0),
            "Station": [station_id] * 25,
            "Last Updated": [datetime.now(ZoneInfo('America/Chicago')).strftime("%Y-%m-%d %H:%M CST/CDT")] * 25
        })
        return df

# Fetch for selected
dfs = [get_noaa_data(buoy) for buoy in selected_buoys]
combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

# === LIVE TIMESTAMP ===
st.caption(f"Data last refreshed: {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S CST/CDT')} | NOAA updates hourly")

# === PLOT 1: MULTI-BUOY SPECTRAL ENERGY ===
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

# === LIVE ANIMATED RESISTIVITY PULSE WITH START/STOP ===
st.subheader("Live MWD Resistivity Pulse (Mud Pulse Telemetry)")

# Session state for button
if 'pulse_running' not in st.session_state:
    st.session_state.pulse_running = False

# Start/Stop button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Start Pulse" if not st.session_state.pulse_running else "Stop Pulse"):
        st.session_state.pulse_running = not st.session_state.pulse_running

# Containers
frame = st.empty()
status = st.empty()

# Run animation only if ON
if st.session_state.pulse_running:
    for i in range(100):
        t = np.linspace(0, 2, 200)
        pulse_time = (t - (i * 0.02)) % 2
        amplitude = np.sin(2 * np.pi * 5 * pulse_time) * np.exp(-pulse_time * 3)
        noise = np.random.normal(0, 0.05, len(t))
        signal = amplitude + noise
        ping_df = pd.DataFrame({"Time (s)": t, "Amplitude": signal})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ping_df["Time (s)"], y=ping_df["Amplitude"],
            mode='lines', line=dict(color='cyan', width=3)
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            height=300,
            template="plotly_dark",
            showlegend=False,
            xaxis_title="Time (s)",
            yaxis_title="Signal Strength"
        )
        frame.plotly_chart(fig, use_container_width=True)
        
        if 30 < i < 70:
            status.success(f"PULSE DETECTED @ {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S CST/CDT')}")
        else:
            status.info("Waiting for next pulse...")
        
        time.sleep(0.1)
        
        # Stop if button toggled
        if not st.session_state.pulse_running:
            break
else:
    # Show static pulse when stopped
    t = np.linspace(0, 2, 200)
    ping = np.sin(2 * np.pi * 5 * t) * np.exp(-t*3)
    ping_df = pd.DataFrame({"Time (s)": t, "Amplitude": ping})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ping_df["Time (s)"], y=ping_df["Amplitude"],
                             mode='lines', line=dict(color='cyan', width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(height=300, template="plotly_dark", showlegend=False,
                      xaxis_title="Time (s)", yaxis_title="Signal Strength")
    frame.plotly_chart(fig, use_container_width=True)
    status.info("Pulse stopped. Click 'Start Pulse' to begin.")

# === DYNAMIC MAP: ALWAYS VISIBLE ===
st.subheader("Selected Buoy Locations (Gulf of Mexico)")
m = folium.Map(location=[25.0, -90.0], zoom_start=5, tiles="CartoDB dark_matter")

buoy_coords = {
    "42001": [25.933, -86.733],
    "42002": [27.933, -88.233],
    "42003": [28.933, -84.733],
    "42004": [26.933, -87.733],
    "42005": [27.933, -88.233],
    "42007": [25.933, -86.733],
    "42008": [27.933, -88.233],
    "42009": [28.933, -84.733],
    "42010": [26.933, -87.733],
    "42012": [27.933, -88.233],
    "42013": [28.933, -84.733],
    "42019": [26.933, -87.733],
    "42020": [27.933, -88.233],
    "42022": [28.933, -84.733],
    "42035": [26.933, -87.733],
    "42039": [27.933, -88.233],
    "42040": [28.933, -84.733],
    "42045": [26.933, -87.733],
    "42046": [27.933, -88.233],
    "42047": [28.933, -84.733]
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

# === CALL TO ACTION ===
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis position with MRE Consulting
""")
