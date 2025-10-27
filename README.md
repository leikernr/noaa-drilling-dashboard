# noaa-drilling-dashboard

# Submarine Sonar to Subsea Sensors Dashboard

**Live Demo**: (https://leikernr-noaa-drilling-dashboard-noaa-q7uglc.streamlit.app/))

Built by [leikernr], STS2(SS) U.S. Navy Veteran | 14 Years MWD Engineer | BS Technical Management + CS (in progress)

## Built By  
**Leike**  
- **U.S. Navy Veteran** – STS2(SS), 5 years submarine sonar operations  
- **14 Years MWD Engineer** – Real-time drilling telemetry (gamma, resistivity, torque)  
- **BS Technical Management** + **BS Computer Science (in progress)**  

---

## The Bridge: From Sonar to Sensors

| Submarine Sonar | → | MWD Drilling | → | Energy Tech |
|-----------------|----|--------------|-----|-------------|
| Real-time DSP under pressure | → | Gamma/resistivity telemetry | → | IIoT + CTRM dashboards |
| 24/7 mission uptime | → | Rig sensor reliability | → | Digital twin POCs |

---

## Features

- **Live Data Ingest**: NOAA NDBC Buoy 42001 (Gulf of Mexico) via public API  
- **Interactive Charts**:
- Wave spectral energy → Analogous to **MWD gamma ray intensity**  
- Simulated **resistivity pulse** (like active sonar ping)  
- **Geospatial Map**: Buoy + sample rig location (Folium)  
- **Auto-refresh**: Caches data every 10 minutes  
- **Mobile-friendly**: Works on phone, tablet, or rig office

---

## Tech Stack

Python
├── Streamlit          → Web dashboard
├── Pandas             → Data parsing
├── Plotly             → Interactive charts
├── Folium             → Map visualization
├── Requests           → NOAA API
└── Streamlit-Folium   → Map in Streamlit

**Contact**: www.linkedin.com/in/nicholas-leiker-50686755 | Targeting Technical Analyst at MRE
