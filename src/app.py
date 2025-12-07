import streamlit as st
import pandas as pd
import sys
import os

# Add root to path so we can import src
sys.path.append(os.getcwd())

from src.inference import Predictor
from config import DATASET_PATH, UI_PAGE_TITLE, UI_LAYOUT

st.set_page_config(page_title=UI_PAGE_TITLE, layout=UI_LAYOUT)

# Custom CSS for better responsiveness
st.markdown("""
<style>
    /* Table responsiveness */
    .stDataFrame {
        width: 100%;
    }
    .stDataFrame table {
        width: 100% !important;
    }
    .stDataFrame td, .stDataFrame th {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 400px;
        overflow-wrap: break-word !important;
    }
    
    /* Metric styling with auto-sizing */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        min-height: 80px;
    }
    .stMetric label {
        font-size: 14px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 20px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        font-size: 12px;
    }
    
    /* Column text wrapping */
    div[data-testid="column"] {
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return Predictor()

@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)

def main():
    st.title("🚀 AI Appointer Assist")
    st.markdown("AI-powered system for recommending next-best assignments based on career history.")
    
    # Load Resources
    with st.spinner("Loading AI Models..."):
        predictor = load_predictor()
        df = load_data()
        
    # Navigation
    mode = st.sidebar.radio("Mode", ["Employee Lookup", "Simulation", "Billet Lookup"])
    
    # Cache control
    if st.sidebar.button("🔄 Clear Cache & Reload"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    
    # Rank Flexibility Control (Global)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Constraint Settings")
    
    col_up, col_down = st.sidebar.columns(2)
    
    with col_up:
        rank_flex_up = st.slider(
            "⬆️ Promotion",
            min_value=0,
            max_value=3,
            value=0,
            help="Allow roles N ranks ABOVE current rank"
        )
    
    with col_down:
        rank_flex_down = st.slider(
            "⬇️ Demotion",
            min_value=0,
            max_value=3,
            value=0,
            help="Allow roles N ranks BELOW current rank"
        )
    
    if rank_flex_up == 0 and rank_flex_down == 0:
        st.sidebar.info("🔒 **Strict**: Exact rank only")
    elif rank_flex_up > 0 and rank_flex_down == 0:
        st.sidebar.info(f"⬆️ **Promotion-focused**: Up to +{rank_flex_up} ranks")
    elif rank_flex_up == 0 and rank_flex_down > 0:
        st.sidebar.warning(f"⬇️ **Demotion allowed**: Down to -{rank_flex_down} ranks")
    else:
        st.sidebar.warning(f"🎨 **Flexible**: +{rank_flex_up}/-{rank_flex_down} ranks")
    
    if mode == "Employee Lookup":
        st.header("Search Officer Record")
        emp_id = st.number_input("Enter Employee ID", min_value=1, value=200001)
        
        # Find Employee
        record = df[df['Employee_ID'] == emp_id]
        
        if not record.empty:
            row = record.iloc[0]
            
            # Display Profile
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rank", row['Rank'])
                st.metric("Branch", row['Branch'])
            with col2:
                st.metric("Current Role", row['current_appointment'])
                st.metric("Pool", row['Pool'])
            with col3:
                st.metric("Entry Type", row['Entry_type'])
                st.text(f"Appointed Since: {row.get('appointed_since', 'Unknown')}")
                
            with st.expander("View History"):
                st.text_area("Appointment History", row['Appointment_history'], height=100)
                st.text_area("Training History", row['Training_history'], height=100)
                
            # Prediction
            if st.button("🔮 Predict Next Appointment"):
                with st.spinner("Analyzing Career Trajectory..."):
                    try:
                        results = predictor.predict(record, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                        
                        st.subheader("Top Recommended Appointments")
                        st.dataframe(
                            results.style.format({"Confidence": "{:.1%}"})
                                   .background_gradient(subset=["Confidence"], cmap="Greens"),
                            use_container_width=True,
                            column_config={
                                "Prediction": st.column_config.TextColumn("Prediction", width="medium"),
                                "Explanation": st.column_config.TextColumn("Explanation", width="large")
                            }
                        )
                        
                        best_role = results.iloc[0]['Prediction']
                        confidence = results.iloc[0]['Confidence']
                        
                        if confidence < 0.2:
                            st.warning("⚠️ Low Confidence Prediction. Manual Review Recommended.")
                        else:
                            st.success(f"Top Recommendation: **{best_role}**")
                            
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                        
        else:
            st.error("Employee ID not found.")
            
    elif mode == "Simulation":
        st.header("🎮 Career Simulation Playground")
        st.markdown("Explore hypothetical career scenarios with intelligent constraints.")
        
        col1, col2 = st.columns(2)
        with col1:
            rank = st.selectbox("Rank", sorted(df['Rank'].unique()))
            branch = st.selectbox("Branch", sorted(df['Branch'].unique()))
            pool = st.selectbox("Pool", sorted(df['Pool'].unique()))
            
        with col2:
            entry = st.selectbox("Entry Type", sorted(df['Entry_type'].unique()))
            
            # Smart role selection: Filter roles by selected rank and branch
            # Get roles from constraints that match the selected rank and branch
            import json
            with open('models/all_constraints.json') as f:
                constraints = json.load(f)
            
            # Find roles that allow this rank and branch
            valid_roles = []
            for role_name, role_const in constraints.items():
                allowed_ranks = role_const.get('ranks', [])
                allowed_branches = role_const.get('branches', [])
                
                # Check if role is valid for selected rank and branch
                if rank in allowed_ranks and branch in allowed_branches:
                    valid_roles.append(role_name)
            
            if valid_roles:
                valid_roles = sorted(valid_roles)
                last_role = st.selectbox(
                    "Current Role",
                    valid_roles,
                    help=f"Showing roles valid for {rank} in {branch} branch"
                )
            else:
                st.warning(f"No roles found for {rank} in {branch} branch. Using fallback.")
                last_role = "Generic Officer Role"
        
        # Auto-calculate years of service based on rank
        rank_to_years = {
            'Ensign': 2,
            'Lieutenant (jg)': 4,
            'Lieutenant': 7,
            'Lieutenant Commander': 11,
            'Commander': 16,
            'Captain': 22,
            'Commodore': 28,
            'Rear Admiral': 32
        }
        years_service = rank_to_years.get(rank, 5)
        
        st.info(f"📊 Auto-calculated: ~{years_service} years of service for {rank}")
        
        if st.button("🔮 Run Simulation"):
            # Create realistic dummy data
            dummy_data = {
                'Employee_ID': [999999],
                'Rank': [rank],
                'Branch': [branch],
                'Pool': [pool],
                'Entry_type': [entry],
                'Appointment_history': [f"{last_role} (01 JAN 2300 - )"],
                'Training_history': [f"Basic Training (01 JAN 2290 - 01 FEB 2290), Advanced {branch} Course (01 JAN 2295 - 01 JUN 2295)"], 
                'Promotion_history': [f"{rank} (01 JAN 2300 - )"],
                'current_appointment': [last_role],
                'appointed_since': ["01/01/2300"]
            }
            
            dummy_df = pd.DataFrame(dummy_data)
            
            # Predict with current rank_flexibility setting
            with st.spinner("Simulating career trajectory..."):
                try:
                    results = predictor.predict(dummy_df, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                    
                    st.subheader("🎯 Simulation Results")
                    st.dataframe(
                        results.style.format({"Confidence": "{:.1%}"})
                               .background_gradient(subset=["Confidence"], cmap="Blues"),
                        use_container_width=True,
                        column_config={
                            "Prediction": st.column_config.TextColumn("Prediction", width="medium"),
                            "Explanation": st.column_config.TextColumn("Explanation", width="large")
                        }
                    )
                    
                    best_role = results.iloc[0]['Prediction']
                    confidence = results.iloc[0]['Confidence']
                    
                    if confidence < 0.15:
                        st.warning("⚠️ Low confidence. Try adjusting Rank Flexibility slider.")
                    else:
                        st.success(f"✨ Top Prediction: **{best_role}** ({confidence:.1%} confidence)")
                        
                except Exception as e:
                    st.error(f"Simulation Error: {e}")
                    st.info("Try adjusting the Rank Flexibility slider or selecting a different role.")
            
    elif mode == "Billet Lookup":
        st.header("Find Candidates for Role")
        st.markdown("Reverse search: Select a target role to find the best fit officers.")
        
        all_roles = sorted(predictor.target_encoder.classes_)
        target_role = st.selectbox("Target Appointment", all_roles)
        
        # Show constraints for this role if available
        constraints = predictor.constraints.get(target_role, {})
        allowed_branches = constraints.get('branches', [])
        allowed_ranks = constraints.get('ranks', [])
        
        if allowed_branches:
            st.info(f"Typically held by: **{', '.join(allowed_branches)}** Branch")
        if allowed_ranks:
            st.info(f"Typically held by Ranks: **{', '.join(allowed_ranks)}**")
            
        if st.button("Find Top Candidates"):
            with st.spinner(f"Scanning workforce for '{target_role}'..."):
                # Use the new predict_for_role method
                candidates = df.copy()
                
                # Pre-filter by branch to reduce computation
                if allowed_branches:
                    candidates = candidates[candidates['Branch'].isin(allowed_branches)]
                
                if candidates.empty:
                    st.warning("No candidates found matching Branch requirements.")
                else:
                    # Get confidence for this specific role across all candidates
                    match_df = predictor.predict_for_role(candidates, target_role, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                    
                    if not match_df.empty:
                        # Limit to top 20
                        match_df = match_df.head(20)
                        
                        st.success(f"Found {len(match_df)} recommended candidates.")
                        st.dataframe(
                            match_df.style.format({"Confidence": "{:.1%}"})
                                    .background_gradient(subset=["Confidence"], cmap="Greens"),
                            width="stretch",
                            column_config={
                                "Employee_ID": st.column_config.NumberColumn("Employee ID", format="%d"),
                                "Name": st.column_config.TextColumn("Name", width="medium"),
                                "Rank": st.column_config.TextColumn("Rank", width="small"),
                                "Branch": st.column_config.TextColumn("Branch", width="medium"),
                                "Current_Role": st.column_config.TextColumn("Current Role", width="medium"),
                                "Explanation": st.column_config.TextColumn("Reasoning", width="large")
                            }
                        )
                    else:
                        st.warning("No suitable candidates found for this role.")

if __name__ == "__main__":
    main()
