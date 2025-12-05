import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WHO Life Expectancy Dashboard",
    page_icon="🌐",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING & LOGIC
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Ensure this path matches your folder structure and the pickle contains the NEW model/scaler
        with open(r'model/lifeexp_linreg.pkl', 'rb') as f:
            package = pickle.load(f)
        return package['model'], package['scaler']
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'model/lifeexp_linreg.pkl' exists.")
        return None, None

model, scaler = load_artifacts()

@st.cache_data
def load_data():
    # Replace with the actual path to your CSV file
    df = pd.read_csv(r"data/Life-Expectancy-Data-Updated.csv") 
    return df

def predict_life_expectancy(bmi, hiv, gdp, schooling, vaccine):
    """Helper function to scale input and predict using the new 5 features"""
    if model and scaler:
        # Create DF with exact training column names
        input_data = pd.DataFrame({
            'BMI': [bmi],
            'Incidents_HIV': [hiv],
            'GDP_per_capita': [gdp],
            'Schooling': [schooling],
            'vaccine_index': [vaccine]
        })
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Convert back to DF to handle feature names for the model
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)
        
        # Predict
        return model.predict(input_scaled_df)[0]
    return 0

# -----------------------------------------------------------------------------
# 3. NAVIGATION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_home():
    st.session_state.page = 'home'

# Initialize session state for sliders if not exists (for preset buttons)
default_values = {
    'val_bmi': 40.0,
    'val_hiv': 1.0,
    'val_gdp': 5000.0,
    'val_schooling': 10.0,
    'val_vaccine': 80.0
}
for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -----------------------------------------------------------------------------
# 4. VIEW: HOME PAGE (Dashboard Mode)
# -----------------------------------------------------------------------------
if st.session_state.page == 'home':
    # 1. Header & Title
    st.title("🌐 WHO Life Expectancy Dashboard")
    st.markdown("### 🌍 Global Health Overview (2015 Snapshot)")
    
    # Load data for the dashboard view
    df = load_data()
    
    # Filter for the latest year to show "current" stats
    latest_year = df['Year'].max()
    df_recent = df[df['Year'] == latest_year].copy() # .copy() prevents warnings

    # -------------------------------------------------------------------------
    # INSIGHT 1: KEY PERFORMANCE INDICATORS (KPIs)
    # -------------------------------------------------------------------------
    # Calculate metrics
    global_avg_le = df_recent['Life_expectancy'].mean()
    global_avg_gdp = df_recent['GDP_per_capita'].mean()
    highest_le_country = df_recent.loc[df_recent['Life_expectancy'].idxmax()]
    
    # Display Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            label="Global Avg Life Expectancy",
            value=f"{global_avg_le:.1f} Years",
            delta="2015 Data"
        )
        
    with kpi2:
        st.metric(
            label="Global Avg GDP",
            value=f"${global_avg_gdp:,.0f}",
            delta_color="off"
        )
        
    with kpi3:
        st.metric(
            label="Highest Life Expectancy",
            value=f"{highest_le_country['Life_expectancy']:.1f} Years",
            delta=highest_le_country['Country']
        )

    with kpi4:
        # Simple count of countries analyzed
        st.metric(
            label="Countries Analyzed",
            value=len(df_recent['Country'].unique())
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # INSIGHT 2: THE INEQUALITY GAP (Leaderboards)
    # -------------------------------------------------------------------------
    col_lead1, col_lead2, col_viz = st.columns([1, 1, 2])

    with col_lead1:
        st.subheader("🏆 Top 5 Countries")
        top_5 = df_recent.nlargest(5, 'Life_expectancy')[['Country', 'Life_expectancy']]
        st.dataframe(top_5, hide_index=True, use_container_width=True)

    with col_lead2:
        st.subheader("⚠️ Bottom 5 Countries")
        bottom_5 = df_recent.nsmallest(5, 'Life_expectancy')[['Country', 'Life_expectancy']]
        st.dataframe(bottom_5, hide_index=True, use_container_width=True)

    # -------------------------------------------------------------------------
    # INSIGHT 3: CORRELATION SNAPSHOT
    # -------------------------------------------------------------------------
    with col_viz:
        st.subheader("💰 Wealth vs. Health")
        
        # --- FIX: Create a readable 'Status' column from the encoded one ---
        # If Economy_status_Developed is 1, label 'Developed', else 'Developing'
        if 'Economy_status_Developed' in df_recent.columns:
            df_recent['Status'] = df_recent['Economy_status_Developed'].apply(
                lambda x: 'Developed' if x == 1 else 'Developing'
            )
        else:
            df_recent['Status'] = 'Unknown'

        # Now the scatter plot will work
        fig_mini = px.scatter(
            df_recent,
            x="GDP_per_capita",
            y="Life_expectancy",
            size="Population_mln",
            color="Status", # This now exists!
            hover_name="Country",
            log_x=True, 
            height=300,
            title="Correlation: GDP vs Life Expectancy (2015)"
        )
        fig_mini.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mini, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------------
    # NAVIGATION CARDS
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # ROW 1: PREDICTIVE & ANALYTICAL TOOLS
    # -------------------------------------------------------------------------
    st.header("🛠️ Predictive Models")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔬 Policy Simulation")
        st.write("Adjust BMI, GDP, and Schooling to see how they impact life expectancy.")
        if st.button("Launch Simulation", type="primary", use_container_width=True, help='What-If Scenarios'):
            st.session_state.page = "policy_simulation"
            st.rerun()

    with col2:
        st.subheader("🧠 Model Transparency")
        st.write("See the math behind the predictions and feature importance weights.")
        if st.button("View Model Details", type="primary", use_container_width=True):
            st.session_state.page = "research"
            st.rerun()

    with col3:
        st.subheader("📊 Batch Analysis")
        st.write("Upload a CSV dataset to predict life expectancy for hundreds of countries.")
        if st.button("Start Batch Prediction", type="primary", use_container_width=True, help='Big Data Processing'):
            st.session_state.page = "batch_analysis"
            st.rerun()

    # -------------------------------------------------------------------------
    # ROW 2: EXPLORATORY DATA TOOLS
    # -------------------------------------------------------------------------
    st.markdown("### ") # Add a little breathing room
    st.header("🌍 Data Exploration")

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        # Using a container to group the visual elements
        with st.container(border=True):
            st.subheader("🗺️ Geographic Explorer")
            st.markdown("Interactive choropleth maps to visualize how diseases and economic factors distribute globally.")
            if st.button("Open World Maps", icon="🌎", use_container_width=True):
                st.session_state.page = "geographic_explorer"
                st.rerun()

    with col_ex2:
        with st.container(border=True):
            st.subheader("📉 Trend Analysis")
            st.markdown("Longitudinal time-series data (2000-2015) to track progress or regression in developing nations.")
            if st.button("Open Trend Lines", icon="📈", use_container_width=True):
                st.session_state.page = "trend_analysis"
                st.rerun()

# -----------------------------------------------------------------------------
# 5. VIEW: POLICY SIMULATION (UPDATED INPUTS)
# -----------------------------------------------------------------------------
elif st.session_state.page == 'policy_simulation':
    st.button("← Back to Dashboard", on_click=go_home)
    st.title("🔬 Policy Simulation Engine")
    st.markdown("Adjust the key drivers to simulate Life Expectancy outcomes.")
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        st.subheader("Input Parameters")
        
        # --- Quick Load Profiles ---
        st.caption("Quick Load Profiles:")
        c1, c2 = st.columns(2)
        if c1.button("High Dev Profile 🇯🇵"):
            st.session_state.val_bmi = 55.4
            st.session_state.val_hiv = 0.1
            st.session_state.val_gdp = 42000.0
            st.session_state.val_schooling = 19.2
            st.session_state.val_vaccine = 98.5
            st.rerun()
            
        if c2.button("Low Dev Profile 🇸🇱"):
            st.session_state.val_bmi = 22.1
            st.session_state.val_hiv = 9.4
            st.session_state.val_gdp = 480.0
            st.session_state.val_schooling = 6.5
            st.session_state.val_vaccine = 62.0
            st.rerun()
        
        st.markdown("---")

        # --- Input Sliders (Linked to Session State) ---
        bmi = st.slider("BMI (Average)", 10.0, 70.0, key='val_bmi', help="National Average BMI")
        schooling = st.slider("Schooling (Years)", 0.0, 25.0, key='val_schooling', help="Average years of schooling")
        vaccine = st.slider("Vaccine Index", 0.0, 100.0, key='val_vaccine', help="Composite score of Polio/Diphtheria coverage")
        gdp_capita = st.number_input("GDP Per Capita ($)", 0.0, 120000.0, step=500.0, key='val_gdp')
        hiv_incidents = st.slider("HIV Incidents (per 1k)", 0.0, 20.0, step=0.1, key='val_hiv')

        if st.button("Run Simulation", type="primary", use_container_width=True):
            prediction = predict_life_expectancy(bmi, hiv_incidents, gdp_capita, schooling, vaccine)
            st.session_state['last_pred'] = prediction

    with col_result:
        st.subheader("Simulation Results")
        if 'last_pred' in st.session_state:
            pred = st.session_state['last_pred']
            
            # 1. Show the Raw Number BIG
            st.metric("Predicted Life Expectancy", f"{pred:.1f} Years")

            # 2. Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+delta", 
                value = pred,
                delta = {'reference': 71.0, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [40, 90]},
                    'bar': {'color': "#2E86C1"},
                    'steps': [
                        {'range': [40, 60], 'color': "#ffcccb"},  # Red Zone
                        {'range': [60, 75], 'color': "#ffffd1"},  # Yellow Zone
                        {'range': [75, 90], 'color': "#d1ffcd"}], # Green Zone
                }
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10)) 
            st.plotly_chart(fig, use_container_width=True)
            
            # -------------------------------------------------------------
            # 3. SMARTER AI ANALYSIS (Checks inputs, not just the output)
            # -------------------------------------------------------------
            st.markdown("### 📋 AI Strategic Insights")
            
            # A. Determine Overall Status
            if pred < 60:
                st.error(f"**Status: Critical ({pred:.1f} Years)**\n\nLife expectancy is dangerously low. Immediate policy intervention required.")
            elif pred < 75:
                st.warning(f"**Status: Developing ({pred:.1f} Years)**\n\nConsistent with developing nations. Focused improvements needed to reach global leaders.")
            else:
                st.success(f"**Status: Strong ({pred:.1f} Years)**\n\nMetrics align with high-income, developed nations.")

            # B. Analyze SPECIFIC Drivers (The "Smart" Part)
            bottlenecks = []
            strengths = []

            # Check HIV
            if hiv_incidents > 1.5:
                bottlenecks.append(f"🔴 **HIV Incidence ({hiv_incidents}/1k):** High disease burden is significantly reducing longevity. Prioritize ART coverage.")
            elif hiv_incidents < 0.2:
                strengths.append(f"🟢 **HIV Control:** Very low incidence contributes positively.")

            # Check Schooling
            if schooling < 10.0:
                bottlenecks.append(f"🔴 **Education ({schooling} yrs):** Below global secondary standards. Education correlates strongly with health awareness.")
            elif schooling > 15.0:
                strengths.append(f"🟢 **Education:** High schooling years drive better health outcomes.")

            # Check Vaccine
            if vaccine < 85.0:
                bottlenecks.append(f"🔴 **Immunization ({vaccine}%):** Gaps in vaccine coverage leave population vulnerable to preventable outbreaks.")
            
            # Check GDP
            if gdp_capita < 2000:
                bottlenecks.append(f"🟠 **Economic Factors:** Low GDP per capita limits healthcare infrastructure investment.")

            # C. Render Recommendations
            if bottlenecks:
                st.subheader("⚠️ Priority Interventions")
                for item in bottlenecks:
                    st.markdown(item)
            
            if strengths and pred > 70:
                st.subheader("✅ Key Strengths")
                for item in strengths:
                    st.markdown(item)
            
            if not bottlenecks and not strengths:
                st.info("Metrics are balanced. No extreme outliers detected.")

        else:
            st.info("👈 Select a profile or adjust parameters and click 'Run Simulation'.")

# -----------------------------------------------------------------------------
# 6. VIEW: RESEARCH TOOLS (UPDATED COEFFS)
# -----------------------------------------------------------------------------
elif st.session_state.page == 'research':
    st.button("← Back to Dashboard", on_click=go_home)
    st.title("🧠 Model Intelligence & Coefficients")
    
    if model:
        # Visualizing Coefficients for the 5 new features
        # Ensure order matches training
        feature_names = ['BMI', 'Incidents_HIV', 'GDP_per_capita', 'Schooling', 'vaccine_index']
        
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': model.coef_
        })
        
        intercept = model.intercept_
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Linear Regression Weights")
            st.write("How much does each feature contribute?")
            st.dataframe(coef_df, hide_index=True, use_container_width=True)
            st.info(f"**Model Intercept (β₀):** {intercept:.4f} Years")
        
        with col2:
            st.subheader("Feature Impact")
            fig = px.bar(coef_df, x='Weight', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        st.subheader("🧮 The Mathematical Formula")
        
        # 1. The Generic Formula
        st.latex(r"LifeExp = \beta_0 + \beta_1(BMI) + \beta_2(HIV) + \beta_3(GDP) + \beta_4(School) + \beta_5(Vaccine)")
        
        # 2. The Actual Formula with Numbers (Dynamic)
        st.write("**Actual Calculated Model:**")
        try:
            st.latex(rf"""
            LifeExp = {intercept:.2f} 
            + ({model.coef_[0]:.2f} \times BMI) 
            + ({model.coef_[1]:.2f} \times HIV) 
            + ({model.coef_[2]:.5f} \times GDP)
            + ({model.coef_[3]:.2f} \times Schooling)
            + ({model.coef_[4]:.2f} \times Vaccine)
            """)
        except:
            st.write("Formula rendering error (check coefficient count)")

# -----------------------------------------------------------------------------
# 7. VIEW: BATCH ANALYSIS (UPDATED COLS)
# -----------------------------------------------------------------------------
elif st.session_state.page == 'batch_analysis':
    st.button("← Back to Dashboard", on_click=go_home)
    st.title("📊 Batch Analysis Engine")
    st.write("Upload a CSV dataset to generate predictions for multiple countries simultaneously.")
    
    # Updated requirement string
    req_cols_str = "BMI, Incidents_HIV, GDP_per_capita, Schooling, vaccine_index"
    uploaded_file = st.file_uploader(f"Upload CSV (Must contain: {req_cols_str})", type="csv")
    
    if uploaded_file and model:
        try:
            df = pd.read_csv(uploaded_file)
            
            # 1. VALIDATION: Check if required columns exist (Updated list)
            required_cols = ['BMI', 'Incidents_HIV', 'GDP_per_capita', 'Schooling', 'vaccine_index']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if not missing_cols:
                st.write("Data uploaded successfully. Preview:", df.head(3))
                
                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Processing Model..."):
                        # 2. PREPROCESSING
                        input_data = df[required_cols]
                        
                        # 3. SCALING
                        scaled_array = scaler.transform(input_data)
                        scaled_df = pd.DataFrame(scaled_array, columns=required_cols)
                        
                        # 4. PREDICTION
                        predictions = model.predict(scaled_df)
                        
                        # 5. MERGE
                        df['Predicted_Life_Expectancy'] = np.round(predictions, 2)
                        
                        st.success("✅ Batch Processing Complete!")
                        
                        # 6. VISUALIZATION & OUTPUT
                        col_metrics1, col_metrics2 = st.columns([2, 1])
                        
                        with col_metrics1:
                            st.subheader("Prediction Distribution")
                            fig = px.histogram(
                                df, x="Predicted_Life_Expectancy", nbins=20, 
                                title="Distribution of Predicted Life Expectancy",
                                color_discrete_sequence=['#4CAF50']
                            )
                            fig.update_layout(bargap=0.1)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col_metrics2:
                            st.subheader("Summary Stats")
                            st.write(df['Predicted_Life_Expectancy'].describe())
                        
                        # 7. DOWNLOAD
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predicted Data (CSV)",
                            data=csv,
                            file_name="life_expectancy_predictions.csv",
                            mime="text/csv"
                        )
                        
                        with st.expander("View Full Result Data"):
                            st.dataframe(df)
            else:
                st.error(f"❌ Error: The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
                
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# -----------------------------------------------------------------------------
# 8. VIEW: GEOGRAPHIC EXPLORER (Same as before)
# -----------------------------------------------------------------------------
elif st.session_state.page == 'geographic_explorer':
    st.button("← Back to Dashboard", on_click=go_home)
    st.title("🗺️ Geographic Health Explorer")
    try:
        df = load_data()
        col_controls1, col_controls2 = st.columns([2, 1])
        with col_controls1:
            min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
            selected_year = st.slider("Select Year", min_year, max_year, max_year)
        with col_controls2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if 'Year' in numeric_cols: numeric_cols.remove('Year')
            selected_metric = st.selectbox("Select Indicator", numeric_cols, index=0)

        filtered_df = df[df['Year'] == selected_year]
        st.subheader(f"Global {selected_metric} in {selected_year}")
        fig = px.choropleth(
            filtered_df, locations="Country", locationmode="country names",
            color=selected_metric, hover_name="Country",
            color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth"
        )
        fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading map data: {e}")

# -----------------------------------------------------------------------------
# 9. VIEW: TREND ANALYSIS (Same as before)
# -----------------------------------------------------------------------------
elif st.session_state.page == 'trend_analysis':
    st.button("← Back to Dashboard", on_click=go_home)
    st.title("📉 Longitudinal Trend Analysis (2000-2015)")
    try:
        df = load_data()
        with st.container():
            col_config1, col_config2 = st.columns([1, 1])
            with col_config1:
                all_countries = sorted(df['Country'].unique())
                default_countries = ['Turkiye', 'Germany', 'Afghanistan'] if 'Turkiye' in all_countries else all_countries[:3]
                selected_countries = st.multiselect("Select Countries", all_countries, default=default_countries)
            with col_config2:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if 'Year' in numeric_cols: numeric_cols.remove('Year')
                selected_metric = st.selectbox("Select Indicator", numeric_cols, index=0)

        if selected_countries:
            filtered_df = df[df['Country'].isin(selected_countries)]
            st.subheader(f"Trend: {selected_metric} over Time")
            fig = px.line(filtered_df, x="Year", y=selected_metric, color="Country", markers=True, template="plotly_white")
            fig.update_layout(height=500, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one country.")
    except Exception as e:
        st.error(f"Error loading trend data: {e}")

# Footer
st.divider()
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.markdown("""
    * Developed by Adarsh | © 2025
    * **Data Source:** World Health Organization (WHO)
    * **Model:** Linear Regression (v1.2 - Updated Features)
    * **GitHub:** https://github.com/adarsh-crafts/life-expectancy-modeling
            
    """)
with footer_col2:
    st.caption("""
    **Disclaimer:** This tool is for educational purposes only. 
    Predictions are based on historical data correlations.
    """)