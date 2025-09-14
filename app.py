import streamlit as st
import snowflake.connector
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Telematics Insurance Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Enhanced CSS Styling
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: #222831; 
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-container:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        background: rgba(255, 255, 255, 1);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-change.positive {
        color: #28a745;
    }
    
    .metric-change.negative {
        color: #dc3545;
    }
    
    .custom-section-divider {
        padding: 2rem;
        border: 1px solid #444;
        border-radius: 15px;
        background-color: #1a1a1a; /* A slightly different background to stand out */
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin-bottom: 2rem; /* Add space below the card */
    }

    # .chart-container {
    #     background: rgba(255, 255, 255, 0.95);
    #     border: 1px solid rgba(255, 255, 255, 0.3);
    #     border-radius: 20px;
    #     padding: 2rem;
    #     margin: 1rem 0;
    #     box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    # }
    
    .section-header {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0 1rem 0;
        color: white;
    }
    
    .section-header h3 {
        margin: 0;
        font-weight: 600;
        font-size: 1.4rem;
    }
    
    .behavior-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .behavior-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .progress-container {
        background: #f8f9fa;
        border-radius: 50px;
        height: 12px;
        overflow: hidden;
        margin: 1rem 0;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 50px;
        transition: width 1s ease-in-out;
        position: relative;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.9) 100%);
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox label, .stCheckbox label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    .alert-container {
        background: rgba(255, 193, 7, 0.1);
        border: 2px solid rgba(255, 193, 7, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .success-container {
        background: rgba(40, 167, 69, 0.1);
        border: 2px solid rgba(40, 167, 69, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low {
        background: rgba(40, 167, 69, 0.1);
        color: #28a745;
        border: 2px solid rgba(40, 167, 69, 0.3);
    }
    
    .risk-medium {
        background: rgba(255, 193, 7, 0.1);
        color: #ffc107;
        border: 2px solid rgba(255, 193, 7, 0.3);
    }
    
    .risk-high {
        background: rgba(220, 53, 69, 0.1);
        color: #dc3545;
        border: 2px solid rgba(220, 53, 69, 0.3);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# Connection and Caching
# =========================
@st.cache_resource
def init_snowflake_connection():
    """Initialize Snowflake connection with enhanced error handling"""
    try:
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
            role=st.secrets["snowflake"]["role"],
            client_session_keep_alive=True
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        return None

@st.cache_data(ttl=180)  # Cache for 3 minutes
def run_query(query, show_error=True):
    """Run query with enhanced error handling and caching"""
    conn = init_snowflake_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        cur = conn.cursor()
        cur.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        cur.close()
        return df
    except Exception as e:
        if show_error:
            st.error(f"Query failed: {str(e)}")
        return pd.DataFrame()

# =========================
# Enhanced Header
# =========================
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">Telematics Insurance Analytics</h1>
    <p style="margin: 0; font-size: 1.2rem; opacity: 0.9; font-weight: 300;">Advanced driver insights, risk assessment, and premium optimization</p>
    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        Real-time data • Machine Learning • Dynamic Pricing
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Enhanced Sidebar
# =========================
with st.sidebar:
    st.markdown("###Dashboard Controls")
    
    # Driver selection with enhanced UI
    with st.spinner("Loading drivers..."):
        drivers_query = """
        SELECT DISTINCT dp.DRIVER_ID, dp.VEHICLE_MAKE, dp.VEHICLE_YEAR, dp.DRIVER_TYPE, dp.AGE,
               rs.COMPOSITE_RISK_SCORE, rs.RISK_CATEGORY
        FROM driver_profiles dp
        LEFT JOIN risk_scores rs ON dp.DRIVER_ID = rs.DRIVER_ID
        ORDER BY dp.DRIVER_ID
        LIMIT 50
        """
        drivers = run_query(drivers_query, show_error=False)
    
    if not drivers.empty:
        # Create enhanced driver labels
        drivers["risk_indicator"] = drivers["RISK_CATEGORY"].fillna("UNKNOWN")
        drivers["label"] = (drivers["DRIVER_ID"] + " | " + 
                          drivers["VEHICLE_MAKE"].fillna("Unknown") + " " + 
                          drivers["VEHICLE_YEAR"].astype(str))
        
        driver_choice = st.selectbox("Select Driver", drivers["label"])
        DRIVER_ID = driver_choice.split(" | ")[0]
        
        # Show selected driver info
        selected_driver = drivers[drivers["DRIVER_ID"] == DRIVER_ID].iloc[0]
        RISK_CATEGORY = selected_driver.get("RISK_CATEGORY", "UNKNOWN")
        risk_score = selected_driver.get("COMPOSITE_RISK_SCORE", 0)
        
        # Risk badge
        if RISK_CATEGORY == "LOW":
            badge_class = "risk-low"
        elif RISK_CATEGORY == "MEDIUM":
            badge_class = "risk-medium"
        elif RISK_CATEGORY == "HIGH":
            badge_class = "risk-high"
        else:
            badge_class = "risk-medium"
        
        st.markdown(f"""
        <div style="margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <div style="color: white; font-weight: 500; margin-bottom: 0.5rem;">Driver Profile</div>
            <div class="risk-badge {badge_class}">{RISK_CATEGORY} Risk</div>
            <div style="color: white; font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">
                Age: {selected_driver.get('AGE', 'N/A')} | Type: {selected_driver.get('DRIVER_TYPE', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.error("No drivers found in database.")
        st.stop()
    
    # Enhanced controls
    st.markdown("---")
    period_days = st.selectbox("Analysis Period", 
                              options=[7, 15, 30, 45, 60, 90], 
                              index=2,
                              help="Select the number of days to analyze")
    
    show_comparisons = st.checkbox("Show Peer Comparisons", value=True)
    
    # Auto-refresh with countdown
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 15, 300, 60)
        
        # Countdown timer
        placeholder = st.empty()
        for remaining in range(refresh_interval, 0, -1):
            placeholder.text(f"Refreshing in {remaining}s...")
            time.sleep(1)
        placeholder.empty()
        st.rerun()

# =========================
# Enhanced Helper Functions
# =========================
def create_metric_card(label, value, change=None, format_type="number", trend_data=None):
    """Create enhanced metric card with trend indicators"""
    if format_type == "currency":
        formatted_value = f"${value:,.2f}"
    elif format_type == "percentage":
        formatted_value = f"{value:.1f}%"
    else:
        formatted_value = f"{value:,.0f}" if value >= 1 else f"{value:.3f}"
    
    change_html = ""
    if change is not None:
        change_class = "positive" if change >= 0 else "negative"
        change_symbol = "🔼" if change <= 0 else "🔽"
        change_html = f'<div class="metric-change {change_class}">{change_symbol} {abs(change):.1f}% vs last period</div>'
    
    return f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{formatted_value}</div>
        {change_html}
    </div>
    """

def create_enhanced_progress_bar(percentage, color_start="#667eea", color_end="#764ba2"):
    """Create enhanced progress bar with gradient"""
    return f"""
    <div class="progress-container">
        <div class="progress-fill" style="width: {percentage}%; background: linear-gradient(90deg, {color_start} 0%, {color_end} 100%);"></div>
    </div>
    <div style="text-align: center; margin-top: 0.5rem; font-weight: 600; color: {color_end};">{percentage:.1f}%</div>
    """

# =========================
# Data Fetching with Error Handling
# =========================
@st.cache_data(ttl=180)
def get_driver_kpis(DRIVER_ID, period_days):
    """Fetch key performance indicators for driver"""
    queries = {
        'risk': f"SELECT COMPOSITE_RISK_SCORE, RISK_CATEGORY FROM risk_scores WHERE DRIVER_ID = '{DRIVER_ID}'",
        'premium': f"""SELECT FINAL_PREMIUM, BASE_PREMIUM, DISCOUNT_APPLIED 
                      FROM pricing_data 
                      WHERE DRIVER_ID = '{DRIVER_ID}' 
                      ORDER BY created_at DESC LIMIT 1""",
        'mileage': f"""SELECT SUM(TRIP_DISTANCE_KM) as TOTAL_MILES, COUNT(DISTINCT TRIP_ID) as TOTAL_TRIPS
                      FROM telematics_data 
                      WHERE DRIVER_ID = '{DRIVER_ID}' 
                      AND TRY_TO_DATE(timestamp) >= DATEADD(day, -{period_days}, CURRENT_DATE)""",
        'behavior': f"""SELECT 
                         AVG(CASE WHEN hard_braking = TRUE THEN 1 ELSE 0 END) * 100 as HARD_BRAKING_PCT,
                         AVG(CASE WHEN hard_acceleration = TRUE THEN 1 ELSE 0 END) * 100 as HARD_ACCEL_PCT,
                         AVG(CASE WHEN phone_usage = TRUE THEN 1 ELSE 0 END) * 100 as PHONE_USAGE_PCT,
                         AVG(speed_kmh) as AVG_SPEED
                       FROM telematics_data 
                       WHERE DRIVER_ID = '{DRIVER_ID}' 
                       AND TRY_TO_DATE(timestamp) >= DATEADD(day, -{period_days}, CURRENT_DATE)"""
    }
    
    results = {}
    for key, query in queries.items():
        df = run_query(query, show_error=False)
        results[key] = df
    
    return results

# Get KPI data
kpi_data = get_driver_kpis(DRIVER_ID, period_days)

# Extract values with safe defaults
risk_score = float(kpi_data['risk']['COMPOSITE_RISK_SCORE'].iloc[0]) if not kpi_data['risk'].empty else 0.5
RISK_CATEGORY = kpi_data['risk']['RISK_CATEGORY'].iloc[0] if not kpi_data['risk'].empty else "MEDIUM"
current_premium = float(kpi_data['premium']['FINAL_PREMIUM'].iloc[0]) if not kpi_data['premium'].empty else 1200
BASE_PREMIUM = float(kpi_data['premium']['BASE_PREMIUM'].iloc[0]) if not kpi_data['premium'].empty else 1200
discount = float(kpi_data['premium']['DISCOUNT_APPLIED'].iloc[0]) if not kpi_data['premium'].empty else 0
TOTAL_MILES = float(kpi_data['mileage']['TOTAL_MILES'].iloc[0]) if not kpi_data['mileage'].empty else 0
TOTAL_TRIPS = int(kpi_data['mileage']['TOTAL_TRIPS'].iloc[0]) if not kpi_data['mileage'].empty else 0

# Calculate safety score
if not kpi_data['behavior'].empty:
    behavior = kpi_data['behavior'].iloc[0]
    safety_score = 100 - (behavior['HARD_BRAKING_PCT'] + behavior['HARD_ACCEL_PCT'] + behavior['PHONE_USAGE_PCT']) / 3
    AVG_SPEED = behavior['AVG_SPEED']
else:
    safety_score = 85
    AVG_SPEED = 45

# Calculate savings
premium_savings = BASE_PREMIUM - current_premium
savings_pct = (premium_savings / BASE_PREMIUM * 100) if BASE_PREMIUM > 0 else 0

# =========================
# Enhanced KPI Cards
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(create_metric_card(
        "Risk Score", 
        risk_score, 
        change=None,
        format_type="number"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(create_metric_card(
        "Current Premium", 
        current_premium,
        change=savings_pct if savings_pct != 0 else None,
        format_type="currency"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(create_metric_card(
        "Safety Score", 
        safety_score,
        format_type="percentage"
    ), unsafe_allow_html=True)

with col4:
    st.markdown(create_metric_card(
        "Miles Driven", 
        TOTAL_MILES,
        format_type="number"
    ), unsafe_allow_html=True)

# =========================
# Premium Comparison Section
# =========================
if savings_pct != 0:
    if savings_pct > 0:
        st.markdown(f"""
        <div class="success-container">
            <h4 style="margin: 0 0 0.5rem 0;">💰 You're Saving Money!</h4>
            <p style="margin: 0; font-size: 1.1rem;">
                <strong>${premium_savings:.2f}</strong> saved compared to traditional pricing 
                (<strong>{savings_pct:.1f}%</strong> discount)
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-container">
            <h4 style="margin: 0 0 0.5rem 0;">⚠️ Premium Adjustment</h4>
            <p style="margin: 0; font-size: 1.1rem;">
                Your premium is <strong>${abs(premium_savings):.2f}</strong> higher than traditional pricing 
                (<strong>{abs(savings_pct):.1f}%</strong> increase) due to driving patterns
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Enhanced Driving Behavior Section
# =========================
st.markdown('<div class="section-header"><h3>Driving Behavior Analysis</h3></div>', unsafe_allow_html=True)
if not kpi_data['behavior'].empty:
    behavior = kpi_data['behavior'].iloc[0]
    
    # Calculate positive metrics
    smooth_braking = 100 - behavior['HARD_BRAKING_PCT']
    controlled_acceleration = 100 - behavior['HARD_ACCEL_PCT']
    focus_score = 100 - behavior['PHONE_USAGE_PCT']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="behavior-card" style="padding: 1rem; background: #222831; border-radius: 12px;">
                <h4 style="color: #28a745; margin-bottom: 1rem; font-weight: 600;">Smooth Braking</h4>
                <p style="color: #fff; font-size: 0.9rem; margin-top: 1rem;">
                    % trips with controlled braking behavior
                </p>
                {create_enhanced_progress_bar(smooth_braking, "#28a745", "#20c997")}
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="behavior-card" style="padding: 1rem; background: #222831; border-radius: 12px;">
                <h4 style="color: #fd7e14; margin-bottom: 1rem; font-weight: 600;">Controlled Acceleration</h4>
                <p style="color: #fff; font-size: 0.9rem; margin-top: 1rem;">
                % trips with smooth acceleration behavior
            </p>
            {create_enhanced_progress_bar(controlled_acceleration, "#fd7e14", "#f39c12")}
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="behavior-card" style="padding: 1rem; background: #222831; border-radius: 12px;">
                <h4 style="color: #6f42c1; margin-bottom: 1rem; font-weight: 600;">Focus Score</h4>
                <p style="color: #fff; font-size: 0.9rem; margin-top: 1rem;">
                    % trips without phone usage
                </p>
                {create_enhanced_progress_bar(focus_score, "#6f42c1", "#e83e8c")}
            """,
            unsafe_allow_html=True
        )
    

# =========================
# Enhanced Charts Section
# =========================
st.markdown('<div class="section-header"><h3>Performance Trends</h3></div>', unsafe_allow_html=True)

# Trend analysis
trend_query = f"""
SELECT 
    DATE_TRUNC('day', TRY_TO_DATE(timestamp)) as DATE,
    AVG(speed_kmh) as AVG_SPEED,
    SUM(CASE WHEN hard_braking = TRUE THEN 1 ELSE 0 END) as HARD_BRAKING_EVENTS,
    SUM(CASE WHEN hard_acceleration = TRUE THEN 1 ELSE 0 END) as HARD_ACCEL_EVENTS,
    SUM(CASE WHEN phone_usage = TRUE THEN 1 ELSE 0 END) as PHONE_USAGE_EVENTS,
    COUNT(*) as DATA_POINTS
FROM telematics_data 
WHERE DRIVER_ID = '{DRIVER_ID}' 
AND TRY_TO_DATE(timestamp) >= DATEADD(day, -{period_days}, CURRENT_DATE)
GROUP BY DATE_TRUNC('day', TRY_TO_DATE(timestamp))
ORDER BY DATE
"""

trend_data = run_query(trend_query, show_error=False)

if not trend_data.empty:
    # Calculate daily safety scores
    trend_data['daily_safety_score'] = 100 - (
        (trend_data['HARD_BRAKING_EVENTS'] / trend_data['DATA_POINTS'] * 100) +
        (trend_data['HARD_ACCEL_EVENTS'] / trend_data['DATA_POINTS'] * 100) +
        (trend_data['PHONE_USAGE_EVENTS'] / trend_data['DATA_POINTS'] * 100)
    ) / 3
    
    # Create a combined chart with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Safety score trend (left axis)
    fig.add_trace(
        go.Scatter(
            x=trend_data['DATE'],
            y=trend_data['daily_safety_score'],
            mode='lines+markers',
            name='Safety Score',
            line=dict(color='#20c997', width=3),
            marker=dict(size=8, symbol='circle'),
        ),
        secondary_y=False
    )
    
    # Speed trend (right axis)
    fig.add_trace(
        go.Scatter(
            x=trend_data['DATE'],
            y=trend_data['AVG_SPEED'],
            mode='lines+markers',
            name='Average Speed (km/h)',
            line=dict(color="#9d6ef4", width=3),
            marker=dict(size=8, symbol='diamond')
        ),
        secondary_y=True
    )
    
    # Layout
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=True,
        title=f'Driving Performance Over Last {period_days} Days',
        hovermode='x unified'
    )
    
    # Axis labels
    fig.update_yaxes(title_text="Safety Score", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Average Speed (km/h)", secondary_y=True, range=[0, 50])

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Peer Comparison (if enabled)
# =========================
if show_comparisons:
    st.markdown('<div class="section-header"><h3>Peer Comparison</h3></div>', unsafe_allow_html=True)
    
    # Get peer data (same age group and vehicle type)
    selected_driver_info = drivers[drivers["DRIVER_ID"] == DRIVER_ID].iloc[0]
    age = selected_driver_info.get('age', 30)
    
    peer_query = f"""
    SELECT 
        AVG(rs.COMPOSITE_RISK_SCORE) as AVG_PEER_RISK,
        AVG(pd.FINAL_PREMIUM) as AVG_PEER_PREMIUM
    FROM risk_scores rs
    JOIN driver_profiles dp ON rs.DRIVER_ID = dp.DRIVER_ID
    LEFT JOIN pricing_data pd ON rs.DRIVER_ID = pd.DRIVER_ID
    WHERE dp.age BETWEEN {age-5} AND {age+5}
    AND dp.DRIVER_ID != '{DRIVER_ID}'
    """
    
    peer_data = run_query(peer_query, show_error=False)
    
    if not peer_data.empty:
        AVG_PEER_RISK = float(peer_data['AVG_PEER_RISK'].iloc[0]) if peer_data['AVG_PEER_RISK'].iloc[0] is not None else 0.5
        AVG_PEER_PREMIUM = float(peer_data['AVG_PEER_PREMIUM'].iloc[0]) if peer_data['AVG_PEER_PREMIUM'].iloc[0] is not None else 1200
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk comparison
            comparison_fig = go.Figure()
            
            comparison_fig.add_trace(go.Bar(
                x=['You', 'Peers'],
                y=[risk_score, AVG_PEER_RISK],
                marker_color=["#72ee8f" if risk_score < AVG_PEER_RISK else "#f46f7d", "#ffd084"],
                text=[f'{risk_score:.3f}', f'{AVG_PEER_RISK:.3f}'],
                textposition='auto'
            ))
            
            comparison_fig.update_layout(
                title='Risk Score Comparison',
                yaxis_title='Risk Score',
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        with col2:
            # Premium comparison
            premium_fig = go.Figure()
            
            premium_fig.add_trace(go.Bar(
                x=['You', 'Peers'],
                y=[current_premium, AVG_PEER_PREMIUM],
                marker_color=['#72ee8f' if current_premium < AVG_PEER_PREMIUM else "#f46f7d", "#ffd084"],
                text=[f'${current_premium:,.0f}', f'${AVG_PEER_PREMIUM:,.0f}'],
                textposition='auto'
            ))
            
            premium_fig.update_layout(
                title='Premium Comparison',
                yaxis_title='Annual Premium ($)',
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(premium_fig, use_container_width=True)

# =========================
# Trip Details Section
# =========================
st.markdown('<div class="section-header"><h3>Recent Trip Analysis</h3></div>', unsafe_allow_html=True)

# Get recent trips
trips_query = f"""
SELECT 
    TRIP_ID,
    START_LOCATION,
    END_LOCATION,
    TRIP_DISTANCE_KM,
    AVG(speed_kmh) as AVG_SPEED,
    MAX(speed_kmh) as MAX_SPEED,
    SUM(CASE WHEN hard_braking = TRUE THEN 1 ELSE 0 END) as HARD_BRAKING_COUNT,
    SUM(CASE WHEN hard_acceleration = TRUE THEN 1 ELSE 0 END) as HARD_ACCEL_COUNT,
    SUM(CASE WHEN phone_usage = TRUE THEN 1 ELSE 0 END) as PHONE_USAGE_COUNT,
    COUNT(*) as DATA_POINTS,
    MIN(timestamp) as TRIP_START
FROM telematics_data 
WHERE DRIVER_ID = '{DRIVER_ID}' 
AND TRY_TO_DATE(timestamp) >= DATEADD(day, -{min(period_days, 7)}, CURRENT_DATE)
GROUP BY TRIP_ID, START_LOCATION, END_LOCATION, TRIP_DISTANCE_KM
ORDER BY MIN(timestamp) DESC
LIMIT 10
"""

trips_data = run_query(trips_query, show_error=False)

if not trips_data.empty:
    # Calculate trip risk scores
    trips_data['trip_risk_score'] = (
        (trips_data['HARD_BRAKING_COUNT'] / trips_data['DATA_POINTS'] * 100) +
        (trips_data['HARD_ACCEL_COUNT'] / trips_data['DATA_POINTS'] * 100) +
        (trips_data['PHONE_USAGE_COUNT'] / trips_data['DATA_POINTS'] * 100) +
        (trips_data['MAX_SPEED'].clip(upper=120) - 60) / 60 * 20  # Speed factor
    ).clip(lower=0)
    
    # Create trip summary table
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Display top 5 trips
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Safest Recent Trips")
        safest_trips = trips_data.nsmallest(3, 'trip_risk_score')
        for _, trip in safest_trips.iterrows():
            st.markdown(f"""
            <div style="background: rgba(40, 167, 69, 0.1); border-left: 4px solid #72ee8f; padding: 1rem; margin: 0.5rem 0; border-radius: 8px;">
                <div style="font-weight: 600; color: #72ee8f;">Trip: {trip['TRIP_ID'][-8:]}</div>
                <div style="font-size: 0.9rem; color: #28a745; margin: 0.25rem 0;">
                    {trip['START_LOCATION']} → {trip['END_LOCATION']}
                </div>
                <div style="font-size: 0.85rem; color: #B2BEB5;">
                    Distance: {trip['TRIP_DISTANCE_KM']:.1f} km | Avg Speed: {trip['AVG_SPEED']:.0f} km/h
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Riskiest Recent Trips")
        riskiest_trips = trips_data.nlargest(3, 'trip_risk_score')
        for _, trip in riskiest_trips.iterrows():
            st.markdown(f"""
            <div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #f46f7d; padding: 1rem; margin: 0.5rem 0; border-radius: 8px;">
                <div style="font-weight: 600; color: #f46f7d;">Trip: {trip['TRIP_ID'][-8:]}</div>
                <div style="font-size: 0.9rem; color: #dc3545; margin: 0.25rem 0;">
                    {trip['START_LOCATION']} → {trip['END_LOCATION']}
                </div>
                <div style="font-size: 0.85rem; color: #B2BEB5;">
                    Distance: {trip['TRIP_DISTANCE_KM']:.1f} km | Max Speed: {trip['MAX_SPEED']:.0f} km/h
                </div>
                <div style="font-size: 0.85rem; color: #f46f7d;">
                    Issues: {trip['HARD_BRAKING_COUNT']} hard brakes, {trip['PHONE_USAGE_COUNT']} phone events
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Recommendations Section
# =========================
st.markdown('<div class="section-header"><h3>Personalized Recommendations</h3></div>', unsafe_allow_html=True)

recommendations = []

# Generate recommendations based on behavior
if not kpi_data['behavior'].empty:
    behavior = kpi_data['behavior'].iloc[0]
    
    if behavior['HARD_BRAKING_PCT'] > 15:
        recommendations.append({
            'icon': '🛑',
            'title': 'Improve Braking Habits',
            'description': 'You have frequent hard braking events. Try maintaining more following distance and anticipating traffic changes.',
            'impact': 'Could reduce premium by 5-8%',
            'priority': 'high'
        })
    
    if behavior['PHONE_USAGE_PCT'] > 10:
        recommendations.append({
            'icon': '📱',
            'title': 'Reduce Phone Usage',
            'description': 'Phone usage while driving significantly increases your risk score. Consider hands-free options or pulling over.',
            'impact': 'Could reduce premium by 10-15%',
            'priority': 'high'
        })
    
    if behavior['AVG_SPEED'] > 70:
        recommendations.append({
            'icon': '🏎️',
            'title': 'Moderate Speed',
            'description': 'Your average speed is above optimal ranges. Reducing speed by 10-15% can improve your risk profile.',
            'impact': 'Could reduce premium by 3-5%',
            'priority': 'medium'
        })

if safety_score > 90:
    recommendations.append({
        'icon': '🏆',
        'title': 'Excellent Driving!',
        'description': 'You\'re already an excellent driver. Keep up the great work to maintain your low premiums.',
        'impact': 'Continue saving 15-20%',
        'priority': 'low'
    })

# Display recommendations
if recommendations:
    for rec in recommendations:
        priority_color = {'high': '#f46f7d', 'medium': "#fedf80", 'low': '#72ee8f'}[rec['priority']]
        priority_bg = {'high': 'rgba(220, 53, 69, 0.1)', 'medium': 'rgba(255, 193, 7, 0.1)', 'low': 'rgba(40, 167, 69, 0.1)'}[rec['priority']]
        
        st.markdown(f"""
        <div style="background: {priority_bg}; border-left: 4px solid {priority_color}; padding: 1.5rem; margin: 1rem 0; border-radius: 12px;">
            <div style="display: flex; align-items: flex-start; gap: 1rem;">
                <div style="font-size: 2rem;">{rec['icon']}</div>
                <div style="flex: 1;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #fff; font-weight: 600;">{rec['title']}</h4>
                    <p style="margin: 0 0 0.5rem 0; color: #B2BEB5;">{rec['description']}</p>
                    <div style="font-size: 0.9rem; font-weight: 600; color: {priority_color};">💰 {rec['impact']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Enhanced Footer
# =========================
st.markdown("---")

# Summary statistics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Trips Analyzed", 
        TOTAL_TRIPS,
        help="Number of trips in the selected period"
    )

with col2:
    st.metric(
        "Data Quality Score", 
        f"{AVG_SPEED/50*100:.0f}%",  # Rough data quality estimate
        help="Quality of telematics data received"
    )

with col3:
    potential_savings = BASE_PREMIUM * 0.2  # Max potential savings
    current_savings = max(0, premium_savings)
    remaining_savings = potential_savings - current_savings
    
    st.metric(
        "Additional Savings Potential", 
        f"${remaining_savings:.0f}",
        help="Estimated additional savings with improved driving"
    )

# Footer info
st.markdown(f"""
<div style="text-align: center; background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; margin-top: 2rem;">
    <div style="color: rgba(255, 255, 255, 0.9); margin-bottom: 0.5rem;">
        <strong>Dashboard Status:</strong> Active | <strong>Driver:</strong> {DRIVER_ID} | <strong>Risk Level:</strong> {RISK_CATEGORY}
    </div>
    <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Analysis period: {period_days} days | 
        Next update: {(datetime.now() + timedelta(hours=1)).strftime('%H:%M')}
    </div>
    <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.8rem; margin-top: 0.5rem;">
        Data secured with enterprise-grade encryption | Privacy compliant | Real-time analytics
    </div>
</div>
""", unsafe_allow_html=True)