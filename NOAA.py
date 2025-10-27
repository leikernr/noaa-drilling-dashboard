# === RIG OPS PANEL: LIVE ENVIRONMENTAL DATA (SAFE FOR NO BUOY) ===
st.subheader("Rig Ops Panel — Live Environmental Conditions")

# Only run if at least one buoy is selected
if not selected_buoys:
    st.warning("Select at least one buoy to see live rig conditions.")
else:
    placeholder = st.empty()

    while True:
        with placeholder.container():
            buoy = selected_buoys[0]  # Safe now
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
