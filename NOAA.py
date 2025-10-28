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

@@ -39,25 +38,17 @@
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
        st.experimental_rerun()

# -------------------------
# === RIGS ===
# === RIGS & BUOYS ===
# -------------------------
real_rigs = [
    {"name": "Neptune TLP", "lat": 27.37, "lon": -89.92},
@@ -66,9 +57,6 @@
    {"name": "Sailfin FPSO", "lat": 27.80, "lon": -90.20}
]

# -------------------------
# === BUOY COORDS ===
# -------------------------
buoy_coords = {
    "42001": [25.933, -86.733],
    "42002": [27.933, -88.233],
@@ -90,7 +78,7 @@
    return R * c * 0.621371

# -------------------------
# === NOAA DATA FUNCTIONS ===
# === NOAA DATA ===
# -------------------------
@st.cache_data(ttl=600)
def get_noaa_data(station_id):
@@ -110,11 +98,7 @@
    except:
        freqs = np.linspace(0.03, 0.40, 25)
        energy = 0.5 + 3 * np.exp(-60 * (freqs - 0.1)**2) + np.random.normal(0, 0.2, 25)
        return pd.DataFrame({
            "Frequency (Hz)": freqs,
            "Spectral Energy (m²/Hz)": energy.clip(0),
            "Station": [station_id] * 25
        })
        return pd.DataFrame({"Frequency (Hz)": freqs, "Spectral Energy (m²/Hz)": energy.clip(0), "Station": [station_id]*25})

@st.cache_data(ttl=600)
def get_realtime_buoy_data(station_id):
@@ -134,17 +118,14 @@
            "MWD": latest.get('MWD', np.nan)
        }
    except:
        return {k: np.nan for k in ["WVHT", "DPD", "WSPD", "GST", "WD", "PRES", "ATMP", "WTMP", "MWD"]}
        return {k: np.nan for k in ["WVHT","DPD","WSPD","GST","WD","PRES","ATMP","WTMP","MWD"]}

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
@@ -154,98 +135,91 @@
                       title=f"Wave Energy — Buoy {selected_buoys[0]}", template="plotly_dark")
    fig1.update_layout(height=350)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Select buoys to view wave energy.")

# -------------------------
# === MWD PULSE SIMULATOR ===
# === WAVE ENERGY VS RIG PROXIMITY ===
# -------------------------
st.subheader("MWD Mud Pulse Telemetry")
st.subheader("Wave Energy Impact on Nearby Rigs")
impact_data = []
for buoy in selected_buoys:
    b_lat, b_lon = buoy_coords[buoy]
    avg_energy = combined_df[combined_df["Station"]==buoy]["Spectral Energy (m²/Hz)"].mean()
    nearest_dist = min([haversine(b_lat,b_lon,r["lat"],r["lon"]) for r in real_rigs])
    impact_data.append({"Buoy": buoy, "Avg Energy": round(avg_energy,2), "Nearest Rig (mi)": round(nearest_dist,1)})
impact_df = pd.DataFrame(impact_data)
if not impact_df.empty:
    fig2 = px.scatter(impact_df, x="Nearest Rig (mi)", y="Avg Energy", hover_data=["Buoy"],
                      title="Wave Energy vs Rig Proximity", template="plotly_dark")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
    bit_pattern = "1010"
# -------------------------
# === MWD PULSE SIMULATOR (NON-BLOCKING) ===
# -------------------------
st.subheader("MWD Mud Pulse Telemetry")
if 'running' not in st.session_state:
    st.session_state.running = True
if 'mwd_frame' not in st.session_state:
    st.session_state.mwd_frame = 0

col1, col2 = st.columns([1, 5])
col1, col2 = st.columns([1,5])
with col1:
    if 'running' not in st.session_state:
        st.session_state.running = True
    if st.button("Stop" if st.session_state.running else "Start"):
        st.session_state.running = not st.session_state.running

frame = st.empty()
status = st.empty()

def generate_mwd_packet():
    t = np.linspace(0, 4, 400)
    t = np.linspace(0,4,400)
    signal = np.zeros_like(t)
    for i, bit in enumerate(bit_pattern):
        pos = 1.0 + i * 0.5
        mask = (t >= pos) & (t < pos + pulse_width)
        signal[mask] = 0.8 if bit == '1' else -0.8
    signal += np.random.normal(0, noise_level, len(t))
    return pd.DataFrame({"Time (s)": t, "Amplitude": signal})
        pos = 1.0 + i*0.5
        mask = (t>=pos) & (t<pos+pulse_width)
        signal[mask] = 0.8 if bit=='1' else -0.8
    signal += np.random.normal(0,noise_level,len(t))
    return pd.DataFrame({"Time (s)":t,"Amplitude":signal})

packet = generate_mwd_packet()
shift = st.session_state.mwd_frame*0.05
t_shifted = (packet["Time (s)"] - shift) % 4
visible = (t_shifted>=0) & (t_shifted<=2)
df_plot = pd.DataFrame({"Time (s)":t_shifted[visible],"Amplitude":packet["Amplitude"][visible]})
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"], mode='lines', line=dict(color='cyan', width=3)))
fig.add_hline(y=0,line_dash="dot",line_color="gray")
fig.add_vline(x=0,line_dash="dash",line_color="red")
fig.add_vline(x=2,line_dash="dash",line_color="red")
fig.update_layout(height=300,template="plotly_dark",showlegend=False,xaxis_range=[0,2])
frame.plotly_chart(fig,use_container_width=True)
status.info("Running..." if st.session_state.running else "Paused")
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
    st.session_state.mwd_frame = (st.session_state.mwd_frame+1)%80
    st.experimental_rerun()

# -------------------------
# === RIG OPS PANEL ===
# -------------------------
st.subheader("Rig Ops — Live Environmental Conditions")

if not selected_buoys:
    st.info("Select a buoy to view live conditions.")
else:
if selected_buoys:
    buoy = selected_buoys[0]
    data = get_realtime_buoy_data(buoy)

    # Wave height
    wave_height = f"{data['WVHT']:.1f} ft" if pd.notna(data['WVHT']) else "—"
    dom_period = f"{data['DPD']:.1f} s" if pd.notna(data['DPD']) else "—"
    wind_speed = f"{data['WSPD']:.1f} kt" if pd.notna(data['WSPD']) else f"{np.random.uniform(1,5):.1f} kt"
    wind_dir = f"{int(data['WD'])}°" if pd.notna(data['WD']) else f"{np.random.randint(0,360)}°"
    pressure = f"{data['PRES']:.2f} inHg" if pd.notna(data['PRES']) else f"{1015:.2f} inHg"
    pressure = f"{data['PRES']:.2f} inHg" if pd.notna(data['PRES']) else "1015.00 inHg"
    wave_dir = f"{int(data['MWD'])}°" if pd.notna(data['MWD']) else f"{np.random.randint(0,360)}°"
    water_temp = f"{(data['WTMP']*9/5+32):.1f}°F" if pd.notna(data['WTMP']) else "—"
    air_temp = f"{(data['ATMP']*9/5+32):.1f}°F" if pd.notna(data['ATMP']) else "—"
    current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
    current_dir = f"{np.random.randint(0,360)}°"
    water_temp = f"{(data['WTMP']*9/5+32):.1f}°F" if pd.notna(data['WTMP']) else "86.2°F"
    air_temp = f"{(data['ATMP']*9/5+32):.1f}°F" if pd.notna(data['ATMP']) else "84.6°F"
    current_speed = f"{np.random.uniform(0.5,2.0):.1f} kt"
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
@@ -266,7 +240,7 @@
# -------------------------
# === MAP ===
# -------------------------
st.subheader("Gulf Buoy & Rig Map")
st.subheader("Buoy & Rig Locations")
m = folium.Map(location=[27.5, -88.5], zoom_start=6)
for rig in real_rigs:
    folium.Marker([rig["lat"], rig["lon"]], popup=rig["name"], icon=folium.Icon(color='blue')).add_to(m)
@@ -287,4 +261,3 @@




