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

@@ -20,7 +20,7 @@
Built by a submarine veteran with 14 years in oilfield telemetry
""")

# === SIDEBAR: FULL ORIGINAL + MWD CONTROLS ===
# === SIDEBAR ===
with st.sidebar:
    st.header("Gulf Buoy Fleet (Select Up to 10)")
    buoy_options = {
@@ -46,31 +46,31 @@
        "42047 - Eastern Gulf": "42047"
    }
    selected_buoys = st.multiselect(
        "Choose buoys for comparison (Gulf offshore array)",
        "Choose buoys",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=10,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][ formatting=0]
    )

    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = Real-time signal processing in extreme noise  
    - **Submarine Sonar** = Real-time signal processing  
    - **MWD Drilling** = Same math: gamma, resistivity, torque  
    - **Energy Tech** = Multi-source fusion for regional ops  
    - **Energy Tech** = Multi-source fusion for ops  
    """)
    st.info("NOAA 420xx series → Gulf Fleet → Analogous to multi-rig sensor arrays")
    st.info("NOAA 420xx → Gulf Fleet → Multi-rig sensor analogy")

    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.05, 0.5, 0.1, 0.05)
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)

    if st.button("Refresh All Buoys"):
    if st.button("Refresh All"):
        st.cache_data.clear()
        st.success("Data refreshed!")
        st.success("Refreshed!")

# === DATA INGEST: SPECTRAL + REAL-TIME ===
# === DATA INGEST ===
@st.cache_data(ttl=600)
def get_noaa_data(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
@@ -86,17 +86,14 @@
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

@st.cache_data(ttl=600)
@@ -108,47 +105,40 @@
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
        return {k: np.nan for k in ["WVHT", "DPD", "APD", "MWD", "WSPD", "GST", "WD", "ATMP", "WTMP", "PRES"]}
        return {k: np.nan for k in ["WVHT", "DPD", "WSPD", "GST", "WD", "PRES", "ATMP", "WTMP", "MWD"]}

dfs = [get_noaa_data(buoy) for buoy in selected_buoys]
combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0] if dfs else pd.DataFrame()

# === LIVE TIMESTAMP ===
st.caption(f"Data last refreshed: {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S CST/CDT')} | NOAA updates hourly")

# === SPECTRAL PLOT ===
if selected_buoys:
    if len(selected_buoys) > 1:
        fig1 = px.line(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)", color="Station",
                       title="Multi-Buoy Wave Spectral Energy = Analogous to Multi-Rig MWD Gamma Intensity",
                       labels={"Spectral Energy (m²/Hz)": "Energy (m²/Hz)"},
                       title="Wave Spectral Energy (Multi-Buoy)",
                       template="plotly_dark")
    else:
        fig1 = px.area(combined_df, x="Frequency (Hz)", y="Spectral Energy (m²/Hz)",
                       title=f"Wave Spectral Energy for Buoy {selected_buoys[0]} = Analogous to MWD Gamma Ray Intensity",
                       labels={"Spectral Energy (m²/Hz)": "Energy (m²/Hz)"},
                       title=f"Wave Energy — Buoy {selected_buoys[0]}",
                       template="plotly_dark")
    fig1.update_layout(height=400)
    fig1.update_layout(height=350)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Select one or more buoys to view wave spectral energy.")
    st.info("Select buoys to view wave energy.")

# === MWD PULSE SIMULATOR ===
st.subheader("Live MWD Mud Pulse Telemetry (6-Pulse Packet)")
# === MWD SIMULATOR ===
st.subheader("MWD Mud Pulse Telemetry")

if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
    bit_pattern = "1010"
    st.warning("Invalid bit pattern. Using '1010'.")

col1, col2 = st.columns([1, 5])
with col1:
@@ -165,7 +155,7 @@
    signal = np.zeros_like(t)
    for pos in [0.0, 0.5]:
        mask = (t >= pos) & (t < pos + pulse_width * 1.5)
        signal[mask] = 1.0
        signal[mask] = 1. SHORT
    for i, bit in enumerate(bit_pattern):
        pos = 1.0 + i * 0.5
        mask = (t >= pos) & (t < pos + pulse_width)
@@ -187,12 +177,9 @@
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(x=2, line_dash="dash", line_color="red")
        fig.update_layout(height=300, template="plotly_dark", showlegend=False,
                          xaxis_title="Time (s)", yaxis_title="Pressure", xaxis_range=[0, 2])
                          xaxis_range=[0, 2])
        frame.plotly_chart(fig, use_container_width=True)
        if 1.5 < shift < 2.8:
            status.success(f"PACKET DECODED @ {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S')}")
        else:
            status.info("Waiting for packet...")
        status.info("Transmitting...")
        time.sleep(0.1)
else:
    packet = generate_mwd_packet()
@@ -204,79 +191,110 @@
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(x=2, line_dash="dash", line_color="red")
    fig.update_layout(height=300, template="plotly_dark", showlegend=False,
                      xaxis_title="Time (s)", yaxis_title="Pressure", xaxis_range=[0, 2])
                      xaxis_range=[0, 2])
    frame.plotly_chart(fig, use_container_width=True)
    status.info("Simulator paused.")
    status.info("Paused.")

# === RIG OPS PANEL: LIVE ENVIRONMENTAL DATA (SAFE FOR NO BUOY) ===
st.subheader("Rig Ops Panel — Live Environmental Conditions")
# === RIG OPS PANEL: 10 PARAMETERS, SHORT DESCRIPTIONS ===
st.subheader("Rig Ops — Live Environmental Conditions")

if not selected_buoys:
    st.warning("Select at least one buoy to see live rig conditions.")
    st.info("Select a buoy to view live conditions.")
else:
    placeholder = st.empty()

    while True:
        with placeholder.container():
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
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Wave Height", "4.2 ft")
                    st.metric("Wind Speed", "18 kt")
                with col2:
                    st.metric("Wind Gust", "22 kt")
                    st.metric("Air Temp", "82.4°F")
                st.success("**DRILLING WINDOW: OPEN**")

        time.sleep(600)  # 10-minute refresh

# === MAP + REAL RIGS ===
st.subheader("Buoy & Active Rig Locations (Gulf of Mexico)")
    buoy = selected_buoys[0]
    try:
        data = get_realtime_buoy_data(buoy)
        
        # Real NOAA data
        wave_height = f"{data['WVHT']:.1f} ft" if not pd.isna(data['WVHT']) else "—"
        dom_period = f"{data['DPD']:.1f} s" if not pd.isna(data['DPD']) else "—"
        wind_speed = f"{data['WSPD']:.1f} kt" if not pd.isna(data['WSPD']) else "—"
        wind_dir = f"{int(data['WD'])}°" if not pd.isna(data['WD']) else "—"
        pressure = f"{data['PRES']:.2f} inHg" if not pd.isna(data['PRES']) else "—"
        wave_dir = f"{int(data['MWD'])}°" if not pd.isna(data['MWD']) else "—"
        water_temp = f"{(data['WTMP'] * 9/5 + 32):.1f}°F" if not pd.isna(data['WTMP']) else "—"
        air_temp = f"{(data['ATMP'] * 9/5 + 32):.1f}°F" if not pd.isna(data['ATMP']) else "—"

        # Simulated (NOAA doesn't provide)
        current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
        current_dir = f"{np.random.randint(0, 360)}°"
        humidity = f"{np.random.randint(60, 95)}%"
        visibility = f"{np.random.uniform(5, 15):.1f} mi"
        subsurface_temp = f"{(data['WTMP'] * 9/5 + 32 - 5):.1f}°F" if not pd.isna(data['WTMP']) else "—"
        salinity = "35.0 PSU"

        # Nearest rig
        buoy_lat, buoy_lon = buoy_coords[buoy]
        nearest_rig = min(real_rigs, key=lambda r: haversine(buoy_lat, buoy_lon, r["lat"], r["lon"]))
        dist = haversine(buoy_lat, buoy_lon, nearest_rig["lat"], nearest_rig["lon"])

        # === 3-COLUMN METRICS WITH SHORT HELP ===
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wave Height", wave_height, help="Rig motion, BHA run")
            st.metric("Dom. Period", dom_period, help="Wave type (swell/chop)")
            st.metric("Wind Speed", wind_speed, help="DP, crane ops")
            st.metric("Wind Dir", wind_dir, help="Crane swing")
        with col2:
            st.metric("Barometric Pressure", pressure, help="Storm forecast")
            st.metric("Wave Dir", wave_dir, help="Vessel heading")
            st.metric("Sea Temp", water_temp, help="Mud cooling, ROV")
            st.metric("Current Speed", current_speed, help="Riser stress")
        with col3:
            st.metric("Air Temp", air_temp, help="Crew comfort")
            st.metric("Humidity", humidity, help="Fog risk")
            st.metric("Visibility", visibility, help="Helicopter ops")
            st.metric("Nearest Rig", nearest_rig["name"], f"{dist:.0f} mi")

        # Drilling Window
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

    except:
        st.info("Live data unavailable. Using simulation.")
        st.metric("Wave Height", "4.2 ft", help="Rig motion")
        st.metric("Wind Speed", "18 kt", help="DP, crane")
        st.success("DRILLING WINDOW: OPEN")

# === WAVE ENERGY vs RIG PROXIMITY ===
st.subheader("Wave Energy vs Rig Proximity")

if selected_buoys:
    energy_summary = combined_df.groupby("Station").apply(lambda x: x["Spectral Energy (m²/Hz)"].mean()). RESET_INDEX()
    energy_summary.columns = ["Buoy", "Avg Energy"]

    impact_data = []
    for _, row in energy_summary.iterrows():
        if row["Buoy"] in selected_buoys:
            b_lat, b_lon = buoy_coords[row["Buoy"]]
            min_dist = min([haversine(b_lat, b_lon, r["lat"], r["lon"]) for r in real_rigs])
            impact_data.append({
                "Buoy": row["Buoy"],
                "Avg Wave Energy (m²/Hz)": round(row["Avg Energy"], 2),
                "Nearest Rig (mi)": round(min_dist, 1)
            })

    impact_df = pd.DataFrame(impact_data)
    if not impact_df.empty:
        fig2 = px.scatter(impact_df, x="Nearest Rig (mi)", y="Avg Wave Energy (m²/Hz)",
                          hover_data=["Buoy"], title="Wave Energy vs Rig Proximity",
                          template="plotly_dark")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select buoys to view wave energy impact.")

# === MAP ===
st.subheader("Buoy & Rig Locations")
m = folium.Map(location=[25.0, -90.0], zoom_start=5, tiles="CartoDB dark_matter")

buoy_coords = {
@@ -294,33 +312,32 @@
        folium.CircleMarker(
            location=buoy_coords[buoy],
            radius=10,
            popup=f"NOAA Buoy {buoy}<br>Wave Spectra = Acoustic Proxy",
            popup=f"Buoy {buoy}",
            color="cyan",
            fill=True
        ).add_to(m)

# === 4 REAL RIGS (BOEM/BSEE 2025) ===
real_rigs = [
    {"name": "Neptune TLP", "operator": "Chevron", "lat": 27.37, "lon": -89.92, "status": "Active", "prod": "10k bbl/day"},
    {"name": "Thunder Hawk SPAR", "operator": "Murphy Oil", "lat": 28.18, "lon": -88.67, "status": "Active", "prod": "8k bbl/day"},
    {"name": "King's Quay FPS", "operator": "Shell", "lat": 27.75, "lon": -89.25, "status": "Active", "prod": "12k bbl/day"},
    {"name": "Sailfin FPSO", "operator": "Beacon Offshore", "lat": 27.80, "lon": -90.20, "status": "Active", "prod": "7k bbl/day"}
    {"name": "Neptune TLP", "lat": 27.37, "lon": -89.92},
    {"name": "Thunder Hawk SPAR", "lat": 28.18, "lon": -88.67},
    {"name": "King's Quay FPS", "lat": 27.75, "lon": -89.25},
    {"name": "Sailfin FPSO", "lat": 27.80, "lon": -90.20}
]

for rig in real_rigs:
    folium.CircleMarker(
        location=[rig["lat"], rig["lon"]],
        radius=12,
        popup=f"{rig['name']} ({rig['operator']})<br>{rig['status']} | {rig['prod']}<br>Updated: {datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M CST/CDT')}",
        popup=rig["name"],
        color="orange",
        fill=True
    ).add_to(m)

st_folium(m, width=700, height=400)

# === DISTANCE TABLE + INTEGRATION ===
# === DISTANCE TABLE ===
if selected_buoys:
    st.subheader("Buoy-to-Rig Distance Analysis")
    st.subheader("Buoy-to-Rig Distances")
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
@@ -335,20 +352,14 @@
            b_lat, b_lon = buoy_coords[buoy]
            for rig in real_rigs:
                dist = haversine(b_lat, b_lon, rig["lat"], rig["lon"])
                dist_data.append({
                    "Buoy": buoy,
                    "Rig": rig["name"],
                    "Distance (mi)": round(dist, 1),
                    "Operator": rig["operator"]
                })
                dist_data.append({"Buoy": buoy, "Rig": rig["name"], "Distance (mi)": round(dist, 1)})

    dist_df = pd.DataFrame(dist_data)
    if not dist_df.empty:
        st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)
    st.dataframe(dist_df.style.highlight_min(axis=0, subset=["Distance (mi)"]), use_container_width=True)

# === CTA ===
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**  
**Now I'll do it for your rig at 55,000 ft.**  
[Contact Me on LinkedIn](www.linkedin.com/in/nicholas-leiker-50686755) | Seeking analysis position with MRE Consulting

