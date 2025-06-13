import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Real-Time Cash Flow Forecasting Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data Generation Functions
@st.cache_data
def generate_sample_data():
    """Generate comprehensive sample financial data"""
    
    # Date range for historical and forecast data
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now() + timedelta(days=180)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Business units
    business_units = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    
    # Generate Accounts Receivable data
    ar_data = []
    for date in date_range:
        for unit in business_units:
            base_amount = random.uniform(50000, 200000)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
            
            ar_data.append({
                'date': date,
                'business_unit': unit,
                'ar_amount': base_amount * seasonal_factor * weekend_factor,
                'days_outstanding': random.randint(15, 90),
                'customer_type': random.choice(['Enterprise', 'SMB', 'Government'])
            })
    
    ar_df = pd.DataFrame(ar_data)
    
    # Generate Accounts Payable data
    ap_data = []
    for date in date_range:
        for unit in business_units:
            base_amount = random.uniform(30000, 150000)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365 + np.pi/4)
            
            ap_data.append({
                'date': date,
                'business_unit': unit,
                'ap_amount': -base_amount * seasonal_factor,  # Negative for outflows
                'payment_terms': random.choice([30, 45, 60]),
                'vendor_type': random.choice(['Supplier', 'Service', 'Utilities'])
            })
    
    ap_df = pd.DataFrame(ap_data)
    
    # Generate Treasury/Cash data
    treasury_data = []
    cash_balance = 1000000  # Starting balance
    
    for date in date_range:
        daily_change = random.uniform(-50000, 50000)
        cash_balance += daily_change
        
        treasury_data.append({
            'date': date,
            'cash_balance': cash_balance,
            'daily_change': daily_change,
            'interest_income': random.uniform(100, 1000),
            'bank_fees': -random.uniform(50, 500)
        })
    
    treasury_df = pd.DataFrame(treasury_data)
    
    # Generate operational cash flows
    operational_data = []
    for date in date_range:
        for unit in business_units:
            operational_data.append({
                'date': date,
                'business_unit': unit,
                'revenue': random.uniform(80000, 300000),
                'operating_expenses': -random.uniform(40000, 150000),
                'capex': -random.uniform(5000, 50000) if random.random() < 0.1 else 0,
                'tax_payments': -random.uniform(10000, 40000) if date.day == 15 else 0
            })
    
    operational_df = pd.DataFrame(operational_data)
    
    return ar_df, ap_df, treasury_df, operational_df

@st.cache_data
def create_unified_cashflow(ar_df, ap_df, treasury_df, operational_df):
    """Create unified cash flow view"""
    
    # Aggregate by date and business unit
    ar_agg = ar_df.groupby(['date', 'business_unit'])['ar_amount'].sum().reset_index()
    ap_agg = ap_df.groupby(['date', 'business_unit'])['ap_amount'].sum().reset_index()
    op_agg = operational_df.groupby(['date', 'business_unit']).agg({
        'revenue': 'sum',
        'operating_expenses': 'sum',
        'capex': 'sum',
        'tax_payments': 'sum'
    }).reset_index()
    
    # Merge all data sources
    unified_df = ar_agg.merge(ap_agg, on=['date', 'business_unit'], how='outer')
    unified_df = unified_df.merge(op_agg, on=['date', 'business_unit'], how='outer')
    
    # Fill NaN values
    unified_df = unified_df.fillna(0)
    
    # Calculate net cash flow
    unified_df['net_cash_flow'] = (
        unified_df['ar_amount'] + 
        unified_df['ap_amount'] + 
        unified_df['revenue'] + 
        unified_df['operating_expenses'] + 
        unified_df['capex'] + 
        unified_df['tax_payments']
    )
    
    # Add cumulative cash flow
    unified_df = unified_df.sort_values(['business_unit', 'date'])
    unified_df['cumulative_cash_flow'] = unified_df.groupby('business_unit')['net_cash_flow'].cumsum()
    
    return unified_df

def build_ml_model(unified_df):
    """Build machine learning model for cash flow prediction"""
    
    # Prepare features
    df_model = unified_df.copy()
    df_model['day_of_week'] = df_model['date'].dt.dayofweek
    df_model['day_of_month'] = df_model['date'].dt.day
    df_model['month'] = df_model['date'].dt.month
    df_model['quarter'] = df_model['date'].dt.quarter
    
    # Create lag features
    df_model = df_model.sort_values(['business_unit', 'date'])
    for lag in [1, 7, 30]:
        df_model[f'net_cash_flow_lag_{lag}'] = df_model.groupby('business_unit')['net_cash_flow'].shift(lag)
    
    # Rolling averages
    for window in [7, 30]:
        df_model[f'net_cash_flow_ma_{window}'] = df_model.groupby('business_unit')['net_cash_flow'].rolling(window=window).mean().reset_index(0, drop=True)
    
    # Remove rows with NaN values
    df_model = df_model.dropna()
    
    # Prepare features and target
    feature_cols = ['day_of_week', 'day_of_month', 'month', 'quarter'] + \
                   [col for col in df_model.columns if 'lag_' in col or 'ma_' in col]
    
    X = df_model[feature_cols]
    y = df_model['net_cash_flow']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols, df_model

def generate_forecast(model, feature_cols, unified_df, days_ahead=90):
    """Generate cash flow forecast"""
    
    forecast_data = []
    last_date = unified_df['date'].max()
    
    for unit in unified_df['business_unit'].unique():
        unit_data = unified_df[unified_df['business_unit'] == unit].copy()
        unit_data = unit_data.sort_values('date')
        
        for i in range(1, days_ahead + 1):
            forecast_date = last_date + timedelta(days=i)
            
            # Create features for forecast
            features = {
                'day_of_week': forecast_date.dayofweek,
                'day_of_month': forecast_date.day,
                'month': forecast_date.month,
                'quarter': (forecast_date.month - 1) // 3 + 1
            }
            
            # Add lag features (using recent actual data)
            for lag in [1, 7, 30]:
                if len(unit_data) >= lag:
                    features[f'net_cash_flow_lag_{lag}'] = unit_data.iloc[-lag]['net_cash_flow']
                else:
                    features[f'net_cash_flow_lag_{lag}'] = 0
            
            # Add moving averages
            for window in [7, 30]:
                if len(unit_data) >= window:
                    features[f'net_cash_flow_ma_{window}'] = unit_data.tail(window)['net_cash_flow'].mean()
                else:
                    features[f'net_cash_flow_ma_{window}'] = unit_data['net_cash_flow'].mean()
            
            # Make prediction
            X_forecast = pd.DataFrame([features])
            X_forecast = X_forecast.reindex(columns=feature_cols, fill_value=0)
            
            predicted_flow = model.predict(X_forecast)[0]
            
            forecast_data.append({
                'date': forecast_date,
                'business_unit': unit,
                'predicted_cash_flow': predicted_flow,
                'forecast_type': 'ML_Prediction'
            })
            
            # Add to unit_data for next iteration
            new_row = pd.DataFrame({
                'date': [forecast_date],
                'business_unit': [unit],
                'net_cash_flow': [predicted_flow]
            })
            unit_data = pd.concat([unit_data, new_row], ignore_index=True)
    
    return pd.DataFrame(forecast_data)

def detect_outliers(df, column='net_cash_flow'):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def calculate_liquidity_risk(unified_df, treasury_df):
    """Calculate liquidity risk metrics"""
    
    # Current cash position
    current_cash = treasury_df['cash_balance'].iloc[-1]
    
    # Calculate cash burn rate (average daily outflow)
    recent_flows = unified_df[unified_df['date'] >= (datetime.now() - timedelta(days=30))]
    daily_burn = recent_flows[recent_flows['net_cash_flow'] < 0]['net_cash_flow'].mean()
    
    # Days of cash remaining
    if daily_burn < 0:
        days_remaining = current_cash / abs(daily_burn)
    else:
        days_remaining = float('inf')
    
    # Liquidity ratio
    monthly_inflow = recent_flows[recent_flows['net_cash_flow'] > 0]['net_cash_flow'].sum()
    monthly_outflow = abs(recent_flows[recent_flows['net_cash_flow'] < 0]['net_cash_flow'].sum())
    
    liquidity_ratio = monthly_inflow / monthly_outflow if monthly_outflow > 0 else float('inf')
    
    return {
        'current_cash': current_cash,
        'daily_burn_rate': daily_burn,
        'days_remaining': days_remaining,
        'liquidity_ratio': liquidity_ratio
    }

def natural_language_query(query, unified_df):
    """Process natural language queries about financial data"""
    
    query_lower = query.lower()
    
    if 'total cash flow' in query_lower:
        total_flow = unified_df['net_cash_flow'].sum()
        return f"The total cash flow across all business units is ${total_flow:,.2f}"
    
    elif 'average' in query_lower and 'cash flow' in query_lower:
        avg_flow = unified_df['net_cash_flow'].mean()
        return f"The average daily cash flow is ${avg_flow:,.2f}"
    
    elif 'best performing' in query_lower or 'highest' in query_lower:
        best_unit = unified_df.groupby('business_unit')['net_cash_flow'].sum().idxmax()
        best_amount = unified_df.groupby('business_unit')['net_cash_flow'].sum().max()
        return f"The best performing business unit is {best_unit} with total cash flow of ${best_amount:,.2f}"
    
    elif 'worst performing' in query_lower or 'lowest' in query_lower:
        worst_unit = unified_df.groupby('business_unit')['net_cash_flow'].sum().idxmin()
        worst_amount = unified_df.groupby('business_unit')['net_cash_flow'].sum().min()
        return f"The worst performing business unit is {worst_unit} with total cash flow of ${worst_amount:,.2f}"
    
    elif 'trend' in query_lower:
        recent_trend = unified_df.tail(30)['net_cash_flow'].mean()
        older_trend = unified_df.head(30)['net_cash_flow'].mean()
        
        if recent_trend > older_trend:
            return f"Cash flow trend is improving. Recent average: ${recent_trend:,.2f} vs Earlier average: ${older_trend:,.2f}"
        else:
            return f"Cash flow trend is declining. Recent average: ${recent_trend:,.2f} vs Earlier average: ${older_trend:,.2f}"
    
    else:
        return "I can help you analyze total cash flow, averages, best/worst performing units, and trends. Try asking about these topics!"

# Main Application
def main():
    st.markdown('<h1 class="main-header">💰 Real-Time Cash Flow Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Generate sample data
    with st.spinner('Loading financial data...'):
        ar_df, ap_df, treasury_df, operational_df = generate_sample_data()
        unified_df = create_unified_cashflow(ar_df, ap_df, treasury_df, operational_df)
        
        # Build ML model
        model, feature_cols, model_df = build_ml_model(unified_df)
        forecast_df = generate_forecast(model, feature_cols, unified_df)
    
    # Sidebar filters
    st.sidebar.header("📊 Dashboard Controls")
    
    # Business unit filter
    business_units = ['All'] + list(unified_df['business_unit'].unique())
    selected_unit = st.sidebar.selectbox("Select Business Unit", business_units)
    
    # Date range filter
    min_date = unified_df['date'].min().date()
    max_date = unified_df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Time horizon for analysis
    time_horizon = st.sidebar.selectbox(
        "Analysis Time Horizon",
        ["7 Days", "30 Days", "90 Days", "1 Year"]
    )
    
    # Filter data based on selections
    if selected_unit != 'All':
        filtered_df = unified_df[unified_df['business_unit'] == selected_unit]
    else:
        filtered_df = unified_df.copy()
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) & 
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "🔮 Forecasting", "🎯 Scenario Planning", 
        "⚠️ Risk Analysis", "🤖 AI Insights", "📊 Detailed Analytics"
    ])
    
    with tab1:
        st.header("Cash Flow Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_inflow = filtered_df[filtered_df['net_cash_flow'] > 0]['net_cash_flow'].sum()
            st.metric("Total Inflows", f"${total_inflow:,.0f}", delta=f"{total_inflow/1000:.1f}K")
        
        with col2:
            total_outflow = filtered_df[filtered_df['net_cash_flow'] < 0]['net_cash_flow'].sum()
            st.metric("Total Outflows", f"${total_outflow:,.0f}", delta=f"{total_outflow/1000:.1f}K")
        
        with col3:
            net_flow = filtered_df['net_cash_flow'].sum()
            st.metric("Net Cash Flow", f"${net_flow:,.0f}", delta=f"{net_flow/1000:.1f}K")
        
        with col4:
            current_cash = treasury_df['cash_balance'].iloc[-1]
            st.metric("Current Cash Balance", f"${current_cash:,.0f}", delta="Live")
        
        # Cash flow trend chart
        st.subheader("Cash Flow Trends")
        
        daily_flows = filtered_df.groupby('date')['net_cash_flow'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_flows['date'],
            y=daily_flows['net_cash_flow'],
            mode='lines+markers',
            name='Daily Cash Flow',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Daily Cash Flow Trend",
            xaxis_title="Date",
            yaxis_title="Cash Flow ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Business unit performance
        st.subheader("Business Unit Performance")
        
        unit_performance = filtered_df.groupby('business_unit')['net_cash_flow'].sum().reset_index()
        
        fig_bar = px.bar(
            unit_performance,
            x='business_unit',
            y='net_cash_flow',
            title="Cash Flow by Business Unit",
            color='net_cash_flow',
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("Cash Flow Forecasting")
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider("Forecast Period (Days)", 30, 180, 90)
        
        with col2:
            confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        
        # Generate forecast with selected parameters
        forecast_df_custom = generate_forecast(model, feature_cols, unified_df, forecast_days)
        
        # Combine historical and forecast data
        historical_summary = unified_df.groupby('date')['net_cash_flow'].sum().reset_index()
        historical_summary['type'] = 'Historical'
        
        forecast_summary = forecast_df_custom.groupby('date')['predicted_cash_flow'].sum().reset_index()
        forecast_summary.rename(columns={'predicted_cash_flow': 'net_cash_flow'}, inplace=True)
        forecast_summary['type'] = 'Forecast'
        
        combined_data = pd.concat([historical_summary, forecast_summary], ignore_index=True)
        
        # Forecast visualization
        fig_forecast = go.Figure()
        
        # Historical data
        historical_data = combined_data[combined_data['type'] == 'Historical']
        fig_forecast.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['net_cash_flow'],
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast data
        forecast_data = combined_data[combined_data['type'] == 'Forecast']
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['net_cash_flow'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig_forecast.update_layout(
            title="Cash Flow Forecast",
            xaxis_title="Date",
            yaxis_title="Cash Flow ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast summary table
        st.subheader("Forecast Summary")
        
        forecast_summary_stats = forecast_df_custom.groupby('business_unit').agg({
            'predicted_cash_flow': ['sum', 'mean', 'std']
        }).round(2)
        
        forecast_summary_stats.columns = ['Total Forecast', 'Average Daily', 'Volatility']
        st.dataframe(forecast_summary_stats)
    
    with tab3:
        st.header("Scenario Planning & Working Capital Optimization")
        
        # Scenario parameters
        st.subheader("Scenario Modeling")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Optimistic Scenario**")
            revenue_increase = st.slider("Revenue Increase (%)", 0, 50, 20, key="opt_rev")
            cost_decrease = st.slider("Cost Decrease (%)", 0, 30, 15, key="opt_cost")
        
        with col2:
            st.write("**Base Case Scenario**")
            st.write("Current forecast (no changes)")
            base_revenue = 0
            base_cost = 0
        
        with col3:
            st.write("**Pessimistic Scenario**")
            revenue_decrease = st.slider("Revenue Decrease (%)", 0, 30, 15, key="pess_rev")
            cost_increase = st.slider("Cost Increase (%)", 0, 40, 20, key="pess_cost")
        
        # Calculate scenarios
        base_forecast = forecast_df_custom.groupby('date')['predicted_cash_flow'].sum()
        
        optimistic_forecast = base_forecast * (1 + (revenue_increase - cost_decrease) / 100)
        pessimistic_forecast = base_forecast * (1 - (revenue_decrease + cost_increase) / 100)
        
        # Scenario visualization
        fig_scenarios = go.Figure()
        
        fig_scenarios.add_trace(go.Scatter(
            x=base_forecast.index,
            y=optimistic_forecast.values,
            mode='lines',
            name='Optimistic',
            line=dict(color='green', width=2),
            fill='tonexty'
        ))
        
        fig_scenarios.add_trace(go.Scatter(
            x=base_forecast.index,
            y=base_forecast.values,
            mode='lines',
            name='Base Case',
            line=dict(color='blue', width=3)
        ))
        
        fig_scenarios.add_trace(go.Scatter(
            x=base_forecast.index,
            y=pessimistic_forecast.values,
            mode='lines',
            name='Pessimistic',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        fig_scenarios.update_layout(
            title="Cash Flow Scenarios",
            xaxis_title="Date",
            yaxis_title="Cash Flow ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        # Working capital optimization
        st.subheader("Working Capital Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Working Capital Metrics**")
            
            # Calculate DSO (Days Sales Outstanding)
            avg_ar = ar_df['ar_amount'].mean()
            avg_daily_sales = operational_df['revenue'].mean()
            dso = avg_ar / avg_daily_sales if avg_daily_sales > 0 else 0
            
            # Calculate DPO (Days Payable Outstanding)
            avg_ap = abs(ap_df['ap_amount'].mean())
            avg_daily_purchases = abs(operational_df['operating_expenses'].mean())
            dpo = avg_ap / avg_daily_purchases if avg_daily_purchases > 0 else 0
            
            st.metric("Days Sales Outstanding (DSO)", f"{dso:.1f} days")
            st.metric("Days Payable Outstanding (DPO)", f"{dpo:.1f} days")
            st.metric("Cash Conversion Cycle", f"{dso - dpo:.1f} days")
        
        with col2:
            st.write("**Optimization Recommendations**")
            
            if dso > 45:
                st.markdown('<div class="warning-box">⚠️ High DSO detected. Consider improving collection processes.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ DSO is within acceptable range.</div>', unsafe_allow_html=True)
            
            if dpo < 30:
                st.markdown('<div class="warning-box">⚠️ Low DPO. Consider negotiating longer payment terms.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ DPO is optimized.</div>', unsafe_allow_html=True)
    
    with tab4:
        st.header("Risk Analysis & Liquidity Management")
        
        # Calculate liquidity risk
        liquidity_metrics = calculate_liquidity_risk(unified_df, treasury_df)
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Cash", f"${liquidity_metrics['current_cash']:,.0f}")
        
        with col2:
            st.metric("Daily Burn Rate", f"${liquidity_metrics['daily_burn_rate']:,.0f}")
        
        with col3:
            days_remaining = liquidity_metrics['days_remaining']
            if days_remaining == float('inf'):
                st.metric("Days of Cash", "∞")
            else:
                st.metric("Days of Cash", f"{days_remaining:.0f} days")
        
        with col4:
            st.metric("Liquidity Ratio", f"{liquidity_metrics['liquidity_ratio']:.2f}")
        
        # Risk level assessment
        if days_remaining < 30:
            st.markdown('<div class="warning-box">🚨 <strong>HIGH RISK:</strong> Less than 30 days of cash remaining!</div>', unsafe_allow_html=True)
        elif days_remaining < 90:
            st.markdown('<div class="warning-box">⚠️ <strong>MEDIUM RISK:</strong> Less than 90 days of cash remaining.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ <strong>LOW RISK:</strong> Sufficient cash reserves.</div>', unsafe_allow_html=True)
        
        # Outlier analysis
        st.subheader("Outlier Analysis")
        
        outliers = detect_outliers(filtered_df)
        
        if not outliers.empty:
            st.write(f"Found {len(outliers)} outlier transactions:")
            
            fig_outliers = px.scatter(
                filtered_df,
                x='date',
                y='net_cash_flow',
                color='business_unit',
                title="Cash Flow with Outliers Highlighted",
                hover_data=['business_unit']
            )
            
            # Highlight outliers
            fig_outliers.add_trace(go.Scatter(
                x=outliers['date'],
                y=outliers['net_cash_flow'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Outliers'
            ))
            
            st.plotly_chart(fig_outliers, use_container_width=True)
            
            # Outlier details
            st.write("Outlier Details:")
            outlier_summary = outliers[['date', 'business_unit', 'net_cash_flow']].sort_values('date', ascending=False)
            st.dataframe(outlier_summary.head(10))
        else:
            st.write("No significant outliers detected in the selected period.")
        
        # Cash flow volatility
        st.subheader("Cash Flow Volatility Analysis")
        
        volatility_by_unit = filtered_df.groupby('business_unit')['net_cash_flow'].std().reset_index()
        volatility_by_unit.columns = ['Business Unit', 'Volatility ($)']
        
        fig_vol = px.bar(
            volatility_by_unit,
            x='Business Unit',
            y='Volatility ($)',
            title="Cash Flow Volatility by Business Unit",
            color='Volatility ($)',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with tab5:
        st.header("AI-Powered Insights & Natural Language Queries")
        
        # Natural language query interface
        st.subheader("Ask Questions About Your Financial Data")
        
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., 'What is the total cash flow?' or 'Which business unit is performing best?'"
        )
        
        if query:
            response = natural_language_query(query, filtered_df)
            st.write("**AI Response:**")
            st.write(response)
        
        # AI-generated insights
        st.subheader("Automated Insights")
        
        # Generate insights
        insights = []
        
        # Cash flow trend insight
        recent_avg = filtered_df.tail(30)['net_cash_flow'].mean()
        overall_avg = filtered_df['net_cash_flow'].mean()
        
        if recent_avg > overall_avg * 1.1:
            insights.append("📈 **Positive Trend**: Recent cash flows are 10%+ above historical average")
        elif recent_avg < overall_avg * 0.9:
            insights.append("📉 **Concerning Trend**: Recent cash flows are 10%+ below historical average")
        
        # Seasonality insight
        monthly_avg = filtered_df.groupby(filtered_df['date'].dt.month)['net_cash_flow'].mean()
        best_month = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()
        
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        insights.append(f"📅 **Seasonality**: Best performing month is {month_names.get(best_month, best_month)}, worst is {month_names.get(worst_month, worst_month)}")
        
        # Business unit insight
        unit_performance = filtered_df.groupby('business_unit')['net_cash_flow'].sum()
        top_performer = unit_performance.idxmax()
        bottom_performer = unit_performance.idxmin()
        
        insights.append(f"🏆 **Top Performer**: {top_performer} leads in cash generation")
        insights.append(f"⚠️ **Attention Needed**: {bottom_performer} requires focus for improvement")
        
        # Volatility insight
        volatility = filtered_df['net_cash_flow'].std()
        if volatility > filtered_df['net_cash_flow'].mean() * 0.5:
            insights.append("⚡ **High Volatility**: Cash flows show significant variation - consider risk management strategies")
        
        # Display insights
        for insight in insights:
            st.markdown(insight)
        
        # Predictive insights
        st.subheader("Predictive Analytics")
        
        # Feature importance from ML model
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Factors Influencing Cash Flow Predictions"
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab6:
        st.header("Detailed Analytics & Data Explorer")
        
        # Data source breakdown
        st.subheader("Data Source Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # AR aging analysis
            st.write("**Accounts Receivable Aging**")
            ar_aging = ar_df.copy()
            ar_aging['aging_bucket'] = pd.cut(
                ar_aging['days_outstanding'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['0-30 days', '31-60 days', '61-90 days', '90+ days']
            )
            
            aging_summary = ar_aging.groupby('aging_bucket')['ar_amount'].sum().reset_index()
            
            fig_aging = px.pie(
                aging_summary,
                values='ar_amount',
                names='aging_bucket',
                title="AR Aging Distribution"
            )
            
            st.plotly_chart(fig_aging, use_container_width=True)
        
        with col2:
            # AP analysis
            st.write("**Accounts Payable by Vendor Type**")
            ap_summary = ap_df.groupby('vendor_type')['ap_amount'].sum().abs().reset_index()
            
            fig_ap = px.bar(
                ap_summary,
                x='vendor_type',
                y='ap_amount',
                title="AP by Vendor Type"
            )
            
            st.plotly_chart(fig_ap, use_container_width=True)
        
        # Detailed data tables
        st.subheader("Raw Data Explorer")
        
        data_source = st.selectbox(
            "Select Data Source",
            ["Unified Cash Flow", "Accounts Receivable", "Accounts Payable", "Treasury", "Operational"]
        )
        
        if data_source == "Unified Cash Flow":
            st.dataframe(filtered_df.head(100))
        elif data_source == "Accounts Receivable":
            st.dataframe(ar_df.head(100))
        elif data_source == "Accounts Payable":
            st.dataframe(ap_df.head(100))
        elif data_source == "Treasury":
            st.dataframe(treasury_df.head(100))
        elif data_source == "Operational":
            st.dataframe(operational_df.head(100))
        
        # Export functionality
        st.subheader("Data Export")
        
        if st.button("Generate Excel Report"):
            # Create Excel file with multiple sheets
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Unified_Cash_Flow', index=False)
                ar_df.to_excel(writer, sheet_name='Accounts_Receivable', index=False)
                ap_df.to_excel(writer, sheet_name='Accounts_Payable', index=False)
                treasury_df.to_excel(writer, sheet_name='Treasury', index=False)
                forecast_df_custom.to_excel(writer, sheet_name='Forecast', index=False)
            
            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name=f"cash_flow_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
