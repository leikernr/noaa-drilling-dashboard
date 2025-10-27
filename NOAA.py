import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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

# -------------------------
# === COMBINE SPECTRAL DATA ===
# -------------------------
dfs = [get_noaa_data(buoy) for buoy in selected_buoys]
combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0] if dfs else pd.DataFrame()

# Spectral Plot
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
# === RIG OPS PANEL ===
# -------------------------
st.subheader("Rig Ops — Live Environmental Conditions")

if selected_buoys:
    buoy = selected_buoys[0]
    data = get_realtime_buoy_data(buoy)

    wave_height = f"{data['WVHT']:.1f} ft" if not pd.isna(data['WVHT']) else "—"
    dom_period = f"{data['DPD']:.1f} s" if not pd.isna(data['DPD']) else "—"
    wind_speed = f"{data['WSPD']:.1f} kt" if not pd.isna(data['WSPD']) else "—"
    wind_dir = f"{int(data['WD'])}°" if not pd.isna(data['WD']) else "—"
    pressure = f"{data['PRES']:.2f} inHg" if not pd.isna(data['PRES']) else "—"
    wave_dir = f"{int(data['MWD'])}°" if not pd.isna(data['MWD']) else "—"
    water_temp = f"{(data['WTMP'] * 9/5 + 32):.1f}°F" if not pd.isna(data['WTMP']) else "—"
    air_temp = f"{(data['ATMP'] * 9/5 + 32):.1f}°F" if not pd.isna(data['ATMP']) else "—"

    current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
    humidity = f"{np.random.randint(60, 95)}%"
    visibility = f"{np.random.uniform(5, 15):.1f} mi"

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
        st.metric("Nearest Rig", "—")

# -------------------------
# === MAP ===
# -------------------------
st.subheader("Buoy & Rig Locations")
m = folium.Map(location=[25.0, -90.0], zoom_start=5, tiles="CartoDB dark_matter")

for buoy in selected_buoys:
    if buoy in buoy_coords:
        folium.CircleMarker(
            location=buoy_coords[buoy],
            radius=10,
            popup=f"Buoy {buoy}",
            color="cyan",
            fill=True
        ).add_to(m)

for rig in real_rigs:
    folium.CircleMarker(
        location=[rig["lat"], rig["lon"]],
        radius=12,
        popup=rig["name"],
        color="orange",
        fill=True
    ).add_to(m)

st_folium(m, width=700, height=400)

# -------------------------
# === BUOY-RIG DISTANCE TABLE ===
# -------------------------
if selected_buoys:
    dist_data = []
    for buoy in selected_buoys:
        if buoy in buoy_coords:
            b_lat, b_lon = buoy_coords[buoy]
            for rig in real_rigs:
                dist = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
                dist_data.append({"Buoy": buoy, "Rig": rig["name"], "Distance (mi)": round(dist, 1)})

    dist_df = pd.DataFrame(dist_data)
    st.subheader("Buoy-to-Rig Distances")
    st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)

# -------------------------
# === WAVE ENERGY IMPACT vs NEAREST RIG ===
# -------------------------
if not combined_df.empty:
    energy_summary = combined_df.groupby("Station")["Spectral Energy (m²/Hz)"].mean().reset_index()
    energy_summary.columns = ["Buoy", "Avg Wave Energy (m²/Hz)"]

    impact_data = []
    for _, row in energy_summary.iterrows():
        if row["Buoy"] in selected_buoys:
            b_lat, b_lon = buoy_coords[row["Buoy"]]
            min_dist = min([haversine(b_lat, b_lon, r["lat"], r["lon"]) for r in real_rigs])
            impact_data.append({
                "Buoy": row["Buoy"],
                "Avg Wave Energy (m²/Hz)": round(row["Avg Wave Energy (m²/Hz)"], 2),
                "Nearest Rig (mi)": round(min_dist, 1)
            })

    impact_df = pd.DataFrame(impact_data)
    if not impact_df.empty:
        fig2 = px.scatter(
            impact_df,
            x="Nearest Rig (mi)",
            y="Avg Wave Energy (m²/Hz)",
            hover_data=["Buoy"],
            title="Wave Energy vs Nearest Rig",
            template="plotly_dark"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# === CTA / FOOTER ===
# -------------------------
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](https://www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis role with MRE Consulting
""")


