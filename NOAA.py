import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Sonar to Sensors", layout="wide")

st.title("Submarine Sonar to Subsea Sensors")
st.markdown("""
**Real-Time Acoustic Energy Dashboard**  
*From U.S. Navy STS2 sonar processing to MWD drilling optimization*  
Built by a submarine veteran with 14 years in oilfield telemetry
""")

with st.sidebar:
    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = Real-time signal processing in extreme noise  
    - **MWD Drilling** = Same math: gamma, resistivity, torque  
    - **Energy Tech** = Needs leaders who’ve *lived* the data pipeline  
    """)
    st.info("NOAA Buoy 42001 → Gulf of Mexico → Analogous to rig sensor array")

@st.cache_data(ttl=600)
def get_noaa_data():
    url = "https://www.ndbc.noaa.gov/data/realtime2/42001.spec"
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
        return pd.DataFrame(data)
    except:
        st.error("NOAA API down. Using cached data.")
        return pd.DataFrame({"Frequency (Hz)": [0.05, 0.1, 0.15], "Spectral Energy (m²/Hz)": [2.1, 3.4, 1.8]})

df = get_noaa_data()

fig1 = px.area(df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)",
               title="Wave Spectral Energy = Analogous to MWD Gamma Ray Intensity",
               labels={"Spectral Energy (m²/Hz)": "Energy (m²/Hz)"},
               template="plotly_dark")
fig1.update_layout(height=400)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Simulated Active Sonar Ping (Resistivity Pulse)")
t = np.linspace(0, 2, 200)
ping = np.sin(2 * np.pi * 5 * t) * np.exp(-t*3)
ping_df = pd.DataFrame({"Time (s)": t, "Amplitude": ping})

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ping_df["Time (s)"], y=ping_df["Amplitude"],
                          mode='lines', name='Resistivity Pulse', line=dict(color='cyan')))
fig2.update_layout(title="Like Sending a Resistivity Tool Pulse Downhole", height=300, template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Sensor Location Context")
m = folium.Map(location=[25.0, -90.0], zoom_start=6, tiles="CartoDB dark_matter")
folium.CircleMarker(
    location=[25.893, -89.668],
    radius=10,
    popup="NOAA Buoy 42001<br>Wave Spectra = Acoustic Proxy",
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
**This is how I processed sonar at 600 ft below sea.**  
**Now I’ll do it for your rig at 6,000 ft.**  
[Contact Me on LinkedIn](www.linkedin.com/in/nicholas-leiker-50686755) | MRE Consulting Ready
""")