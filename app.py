#Define Clinical Performance Dashboard
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import requests

# Page configuration
st.set_page_config(
    page_title="Clinical Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1f6f8b;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        padding: 1rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f6f8b;
    }
    .top-performer-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .weather-card {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def create_clinical_performance_chart(
    df,
    metric_column,
    title,
    color='#1f6f8b',
    sort_by='Clinic',
    ascending=True
):
    """Create a bar chart for clinical performance metrics"""
    df_copy = df.copy()

    # Ensure metric is numeric
    if df_copy[metric_column].dtype == 'object':
        df_copy[metric_column] = (
            df_copy[metric_column]
            .astype(str)
            .str.replace('%', '')
            .str.strip()
        )
        df_copy[metric_column] = pd.to_numeric(df_copy[metric_column], errors='coerce')

    # Controlled sorting
    df_sorted = df_copy.sort_values(sort_by, ascending=ascending)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_sorted['Clinic'],
        y=df_sorted[metric_column],
        text=[f'{val:.0f}%' for val in df_sorted[metric_column]],
        textposition='outside',
        textfont=dict(
            size=16,
            color='black',
            family='Arial Black'
        ),
        marker=dict(
            color=color,
            line=dict(color='black', width=1.4)
        ),
        hovertemplate='<b>%{x}</b><br>' + metric_column + ': %{y:.0f}%<extra></extra>',
        name=metric_column
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(
                size=26,
                color='black',
                family='Arial Black'
            )
        ),

        xaxis=dict(
            title='',
            tickangle=-45,
            showgrid=False,
            tickfont=dict(
                size=14,
                color='black',
                family='Arial Black'
            )
        ),

        yaxis=dict(
            title='',
            showticklabels=False,
            showgrid=False,
            range=[0, df_sorted[metric_column].max() + 18]
        ),

        height=600,

        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',

        margin=dict(
            l=60,
            r=40,
            t=100,
            b=180
        ),

        showlegend=False,
        hovermode='x unified'
    )

    return fig

def get_weather_forecast():
    """Fetch 7 day weather forecast for New York City using Open-Meteo API (free, no API key needed)"""
    try:
        # New York City coordinates
        latitude = 40.7128
        longitude = -74.0060
        
        # Open-Meteo API endpoint (free, no API key required)
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "forecast_days": 7
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Parse the data
        daily = data['daily']
        forecast = []
        
        # Weather code mapping
        weather_codes = {
            0: "☀️ Clear",
            1: "🌤️ Mainly Clear",
            2: "⛅ Partly Cloudy",
            3: "☁️ Overcast",
            45: "🌫️ Foggy",
            48: "🌫️ Foggy",
            51: "🌦️ Light Drizzle",
            53: "🌧️ Drizzle",
            55: "🌧️ Heavy Drizzle",
            61: "🌧️ Light Rain",
            63: "🌧️ Rain",
            65: "🌧️ Heavy Rain",
            71: "🌨️ Light Snow",
            73: "❄️ Snow",
            75: "❄️ Heavy Snow",
            77: "🌨️ Snow Grains",
            80: "🌦️ Light Showers",
            81: "🌧️ Showers",
            82: "⛈️ Heavy Showers",
            85: "🌨️ Light Snow Showers",
            86: "❄️ Snow Showers",
            95: "⛈️ Thunderstorm",
            96: "⛈️ Thunderstorm with Hail",
            99: "⛈️ Severe Thunderstorm"
        }
        
        for i in range(len(daily['time'])):
            date = datetime.strptime(daily['time'][i], '%Y-%m-%d')
            weather_code = daily['weathercode'][i]
            forecast.append({
                'date': date,
                'day_name': date.strftime('%A'),
                'date_str': date.strftime('%b %d'),
                'temp_max': daily['temperature_2m_max'][i],
                'temp_min': daily['temperature_2m_min'][i],
                'precipitation': daily['precipitation_sum'][i],
                'condition': weather_codes.get(weather_code, "🌤️ Partly Cloudy")
            })
        
        return forecast
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def prepare_dataframe(df):
    """Automatically detect and rename columns"""
    clinic_col = None
    visits_col = None
    new_patients_col = None
    utilization_col = None
    actual_visits_col = None
    visit_target_col = None
    actual_new_patients_col = None
    new_patient_target_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if 'clinic' in col_lower:
            clinic_col = col
        elif 'actual' in col_lower and 'visit' in col_lower and 'new' not in col_lower:
            actual_visits_col = col
        elif 'visit' in col_lower and 'target' in col_lower and 'new' not in col_lower:
            visit_target_col = col
        elif 'actual' in col_lower and 'new' in col_lower and 'patient' in col_lower:
            actual_new_patients_col = col
        elif 'new' in col_lower and 'patient' in col_lower and 'target' in col_lower:
            new_patient_target_col = col
        elif 'visit' in col_lower and 'achieve' in col_lower:
            visits_col = col
        elif 'new' in col_lower and 'patient' in col_lower and 'achieve' in col_lower:
            new_patients_col = col
        elif 'utilization' in col_lower:
            utilization_col = col

    column_mapping = {}
    if clinic_col:
        column_mapping[clinic_col] = 'Clinic'
    if actual_visits_col:
        column_mapping[actual_visits_col] = 'Actual Visits'
    if visit_target_col:
        column_mapping[visit_target_col] = 'Visit Target'
    if actual_new_patients_col:
        column_mapping[actual_new_patients_col] = 'Actual New Patients'
    if new_patient_target_col:
        column_mapping[new_patient_target_col] = 'New Patient Target'
    if visits_col:
        column_mapping[visits_col] = 'Visits Achieved %'
    if new_patients_col:
        column_mapping[new_patients_col] = 'New Patients Achieved %'
    if utilization_col:
        column_mapping[utilization_col] = 'Utilization %'

    df_renamed = df.rename(columns=column_mapping)

    # Convert percentage columns to numeric
    for col in ['Visits Achieved %', 'New Patients Achieved %', 'Utilization %']:
        if col in df_renamed.columns:
            df_renamed[col] = (
                df_renamed[col]
                .astype(str)
                .str.replace('%', '')
                .str.strip()
            )
            df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')

    return df_renamed

# Main App
def main():
    st.title("🏥 Daily Clinical Performance")
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload Clinical Performance Data (CSV)",
        type=['csv']
    )
    
    # Load data only from uploaded file
    if uploaded_file is not None:
        df_clinical_Performance = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.info("👆 Use the file uploader in the sidebar to upload your CSV file.")
        st.stop()
    
    # Prepare dataframe
    df_renamed = prepare_dataframe(df_clinical_Performance)
    
    # Sidebar filters
    st.sidebar.header("🏥 Clinic Filter")
    
    all_clinics = sorted(df_renamed['Clinic'].unique().tolist())
    
    # Add "All Clinics" option
    clinic_options = ['All Clinics'] + all_clinics
    
    selected_clinics = st.sidebar.multiselect(
        "Select Clinic(s)",
        options=clinic_options,
        default=['All Clinics']
    )
    
    # Filter dataframe based on selection
    if 'All Clinics' in selected_clinics or len(selected_clinics) == 0:
        df_filtered = df_renamed.copy()
    else:
        df_filtered = df_renamed[df_renamed['Clinic'].isin(selected_clinics)].copy()
    
    # Show number of clinics selected
    if 'All Clinics' in selected_clinics or len(selected_clinics) == 0:
        st.sidebar.info(f"📊 Showing all {len(all_clinics)} clinics")
    else:
        st.sidebar.info(f"📊 Showing {len(selected_clinics)} clinic(s)")
    
    st.sidebar.header("📊 Chart Options")
    
    sort_order = st.sidebar.radio(
        "Sort bars by:",
        ["Metric Value (Ascending)", "Metric Value (Descending)", "Clinic Name (A-Z)"]
    )
    
    show_data_table = st.sidebar.checkbox("Show Data Table", value=False)
    
    # Overview Section
    st.header("📊 Overview")
    
    # Check if raw data columns exist
    has_raw_data = all(col in df_filtered.columns for col in 
                      ['Actual Visits', 'Visit Target', 'Actual New Patients', 'New Patient Target'])
    
    if has_raw_data:
        total_actual_visits = df_filtered['Actual Visits'].sum()
        total_visit_target = df_filtered['Visit Target'].sum()
        total_actual_new_patients = df_filtered['Actual New Patients'].sum()
        total_new_patient_target = df_filtered['New Patient Target'].sum()
        
        # Calculate achievement percentages
        visits_achievement = (total_actual_visits / total_visit_target * 100) if total_visit_target > 0 else 0
        new_patients_achievement = (total_actual_new_patients / total_new_patient_target * 100) if total_new_patient_target > 0 else 0
        
        # Display metrics in a grid
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "📋 Actual Visits",
                f"{total_actual_visits:,}",
                delta=f"{total_actual_visits - total_visit_target:+,} vs target"
            )
        
        with metric_col2:
            st.metric(
                "🎯 Visit Target",
                f"{total_visit_target:,}",
                delta=f"{visits_achievement:.1f}% achieved"
            )
        
        with metric_col3:
            st.metric(
                "👥 Actual New Patients",
                f"{total_actual_new_patients:,}",
                delta=f"{total_actual_new_patients - total_new_patient_target:+,} vs target"
            )
        
        with metric_col4:
            st.metric(
                "🎯 New Patient Target",
                f"{total_new_patient_target:,}",
                delta=f"{new_patients_achievement:.1f}% achieved"
            )
    else:
        st.info("ℹ️ Raw visit and patient data not found. Upload CSV with 'Actual Visits', 'Visit Target', 'Actual New Patients', and 'New Patient Target' columns.")
    
    # Top and Low Performers in a new row
    if 'Visits Achieved %' in df_filtered.columns and len(df_filtered) > 0:
        st.markdown("---")
        performer_col1, performer_col2 = st.columns(2)
        
        with performer_col1:
            # Top Performer
            top_clinic = df_filtered.loc[df_filtered['Visits Achieved %'].idxmax()]
            
            st.markdown("""
                <div class="top-performer-box">
                    <h3 style="margin: 0; color: #856404;">🏆 Top Performer</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"### {top_clinic['Clinic']}")
            st.markdown(f"**Visits Achievement:** {top_clinic['Visits Achieved %']:.1f}%")
            
            if 'New Patients Achieved %' in df_filtered.columns:
                st.markdown(f"**New Patients:** {top_clinic['New Patients Achieved %']:.1f}%")
            
            if 'Utilization %' in df_filtered.columns:
                st.markdown(f"**Utilization:** {top_clinic['Utilization %']:.1f}%")
        
        with performer_col2:
            # Low Performer
            low_clinic = df_filtered.loc[df_filtered['Visits Achieved %'].idxmin()]
            
            st.markdown("""
                <div style="background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 0.5rem; padding: 1rem; text-align: center;">
                    <h3 style="margin: 0; color: #721c24;">⚠️ Low Performer</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"### {low_clinic['Clinic']}")
            st.markdown(f"**Visits Achievement:** {low_clinic['Visits Achieved %']:.1f}%")
            
            if 'New Patients Achieved %' in df_filtered.columns:
                st.markdown(f"**New Patients:** {low_clinic['New Patients Achieved %']:.1f}%")
            
            if 'Utilization %' in df_filtered.columns:
                st.markdown(f"**Utilization:** {low_clinic['Utilization %']:.1f}%")
    
    st.divider()
    
    
    # Determine sort parameters
    if sort_order == "Metric Value (Ascending)":
        sort_ascending = True
        sort_by_metric = True
    elif sort_order == "Metric Value (Descending)":
        sort_ascending = False
        sort_by_metric = True
    else:
        sort_ascending = True
        sort_by_metric = False
    
    # Tabs for different metrics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visits Achievement",
        "👥 New Patients Achievement",
        "⏱️ Utilization",
        "📋 Detailed Comparison",
        "🌤️ Weather Forecast (NYC)"
    ])
    
    with tab1:
        if 'Visits Achieved %' in df_filtered.columns:
            st.subheader("Visits Achievement by Clinic")
            sort_by = 'Visits Achieved %' if sort_by_metric else 'Clinic'
            fig1 = create_clinical_performance_chart(
                df_filtered,
                'Visits Achieved %',
                'Visits Achievement',
                sort_by=sort_by,
                ascending=sort_ascending
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Show clinic performance breakdown
            if len(df_filtered) > 1:
                col_a, col_b = st.columns(2)
                with col_a:
                    above_benchmark = len(df_filtered[df_filtered['Visits Achieved %'] >= 85])
                    st.metric("Clinics Meeting Benchmark (≥85%)", f"{above_benchmark}/{len(df_filtered)}")
                with col_b:
                    below_benchmark = len(df_filtered[df_filtered['Visits Achieved %'] < 85])
                    st.metric("Clinics Below Benchmark (<85%)", f"{below_benchmark}/{len(df_filtered)}")
        else:
            st.warning("'Visits Achieved %' column not found in data")
    
    with tab2:
        if 'New Patients Achieved %' in df_filtered.columns:
            st.subheader("New Patients Achievement by Clinic")
            sort_by = 'New Patients Achieved %' if sort_by_metric else 'Clinic'
            fig2 = create_clinical_performance_chart(
                df_filtered,
                'New Patients Achieved %',
                'New Patients Achievement',
                sort_by=sort_by,
                ascending=sort_ascending
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show clinic performance breakdown
            if len(df_filtered) > 1:
                col_a, col_b = st.columns(2)
                with col_a:
                    above_benchmark = len(df_filtered[df_filtered['New Patients Achieved %'] >= 85])
                    st.metric("Clinics Meeting Benchmark (≥85%)", f"{above_benchmark}/{len(df_filtered)}")
                with col_b:
                    below_benchmark = len(df_filtered[df_filtered['New Patients Achieved %'] < 85])
                    st.metric("Clinics Below Benchmark (<85%)", f"{below_benchmark}/{len(df_filtered)}")
        else:
            st.warning("'New Patients Achieved %' column not found in data")
    
    with tab3:
        if 'Utilization %' in df_filtered.columns:
            st.subheader("Utilization by Clinic")
            sort_by = 'Utilization %' if sort_by_metric else 'Clinic'
            fig3 = create_clinical_performance_chart(
                df_filtered,
                'Utilization %',
                'Utilization',
                sort_by=sort_by,
                ascending=sort_ascending
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Show utilization breakdown
            if len(df_filtered) > 1:
                col_a, col_b = st.columns(2)
                with col_a:
                    above_benchmark = len(df_filtered[df_filtered['Utilization %'] >= 85])
                    st.metric("Clinics Meeting Benchmark (≥85%)", f"{above_benchmark}/{len(df_filtered)}")
                with col_b:
                    below_benchmark = len(df_filtered[df_filtered['Utilization %'] < 85])
                    st.metric("Clinics Below Benchmark (<85%)", f"{below_benchmark}/{len(df_filtered)}")
        else:
            st.warning("'Utilization %' column not found in data")
    
    with tab4:
        st.subheader("Detailed Clinic Comparison")
        
        # Create comparison table
        comparison_cols = ['Clinic']
        if 'Actual Visits' in df_filtered.columns:
            comparison_cols.append('Actual Visits')
        if 'Visit Target' in df_filtered.columns:
            comparison_cols.append('Visit Target')
        if 'Visits Achieved %' in df_filtered.columns:
            comparison_cols.append('Visits Achieved %')
        if 'Actual New Patients' in df_filtered.columns:
            comparison_cols.append('Actual New Patients')
        if 'New Patient Target' in df_filtered.columns:
            comparison_cols.append('New Patient Target')
        if 'New Patients Achieved %' in df_filtered.columns:
            comparison_cols.append('New Patients Achieved %')
        if 'Utilization %' in df_filtered.columns:
            comparison_cols.append('Utilization %')
        
        df_comparison = df_filtered[comparison_cols].copy()
        
        # Format percentage columns for display
        for col in ['Visits Achieved %', 'New Patients Achieved %', 'Utilization %']:
            if col in df_comparison.columns:
                df_comparison[col] = df_comparison[col].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            df_comparison,
            use_container_width=True,
            hide_index=True
        )
    
    with tab5:
        st.subheader("7 Day Weather Forecast for New York City")
        st.markdown("*Plan your clinic operations and patient visits with weather insights*")
        
        with st.spinner("🌤️ Fetching weather data..."):
            forecast = get_weather_forecast()
        
        if forecast:
            # Display forecast in cards
            cols = st.columns(7)
            
            for i, day in enumerate(forecast):
                with cols[i]:
                    temp_max_c = (day['temp_max'] - 32) * 5 / 9
                    temp_min_c = (day['temp_min'] - 32) * 5 / 9
                    st.markdown(f"""
                        <div class="weather-card">
                            <h4 style="margin: 0;font-size: 0.95rem; font-weight: bold;font-weight: 600; text-align: center;">{day['day_name']}</h3>
                            <p style="margin: 0.5rem 0; font-size: 0.9rem;">{day['date_str']}</p>
                            <div style="font-size: 2rem; margin: 1rem 0;">{day['condition'].split()[0]}</div>
                            <p style="margin: 0; font-weight: bold; font-size: 1.2rem;">
                                {temp_max_c:.0f}°C / {temp_min_c:.0f}°C
                            </p>
                            <p style="margin: 0.5rem 0; font-size: 0.85rem;">
                                {day['condition'].split(' ', 1)[1] if len(day['condition'].split()) > 1 else ''}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
                        
            # Weather insights
            st.subheader("📌 Weather Insights for Clinic Operations")
            
            rainy_days = [day for day in forecast if day['precipitation'] > 0.1]
                        
            insight_cols = st.columns(2)
            
            with insight_cols[0]:
                if rainy_days:
                    st.info(f"🌧️ **{len(rainy_days)} rainy day(s)** expected\n\nConsider patient transportation needs")
                else:
                    st.success("☀️ **No significant rain** expected\n\nGood week for travel")
    
    # Optional data table
    if show_data_table:
        st.divider()
        st.header("📋 Raw Data")
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name='clinical_performance_filtered_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
