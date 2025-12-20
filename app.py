"""Streamlit web interface for NVO Rankings predictions."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from nvo.config import load_config
from nvo.services import run_predictions, run_validation
from nvo.data.processors import build_dataset

st.set_page_config(
    page_title="NVO Rankings Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def get_config():
    return load_config()

cfg = get_config()
HISTORICAL_YEARS = [2022, 2023, 2024, 2025]


@st.cache_data
def load_predictions(years, predict_year, gender):
    results, metrics = run_predictions(
        years, predict_year, cfg.data['files_dir'],
        cfg.model, gender, None, use_cache=True
    )
    if results:
        return pd.DataFrame(list(results.values())), metrics
    return pd.DataFrame(), {}


@st.cache_data
def load_validation(train_years, test_year, gender):
    results, metrics = run_validation(
        train_years, test_year, cfg.data['files_dir'],
        cfg.model, gender, None
    )
    if results:
        return pd.DataFrame(list(results.values())), metrics
    return pd.DataFrame(), {}


@st.cache_data
def load_historical_data(years):
    return build_dataset(years, cfg.data['files_dir'])


def main():
    st.title("🎓 NVO 7th Grade Rankings Predictor")
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔮 Predictions", "✅ Validation"])
    
    with tab1:
        prediction_page()
    
    with tab2:
        validation_page()
    
    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with ❤️ using XGBoost\n\n"
        "[GitHub](https://github.com/zhivko-georgiev/nvo-7mi-klas-rankings)"
    )


def prediction_page():
    st.header("🔮 Predictions")
    
    # Settings in columns at top instead of sidebar
    col1, col2, col3 = st.columns(3)
    with col1:
        train_years = st.multiselect(
            "Training Years",
            options=HISTORICAL_YEARS,
            default=[2022, 2023, 2024]
        )
    with col2:
        max_train = max(train_years) if train_years else 2024
        predict_year = st.selectbox("Predict Year", options=[max_train + 1, max_train + 2])
    with col3:
        gender = st.selectbox("Gender", options=["female", "male"], 
                              format_func=lambda x: "Жени (Female)" if x == "female" else "Мъже (Male)")
    
    if len(train_years) < 2:
        st.error("Select at least 2 training years")
        return
    
    gender_label = "Female" if gender == "female" else "Male"
    
    with st.spinner("Loading predictions..."):
        df, metrics = load_predictions(train_years, predict_year, gender)
    
    if df.empty:
        st.error("No predictions available.")
        return
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Profiles", len(df))
    with col2:
        reliable = metrics.get(f"R1_{gender_label}", {}).get('reliable', 0)
        st.metric("Reliable Predictions", reliable)
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search school or profile", "")
    with col2:
        show_reliable = st.checkbox("Reliable only", value=False)
    
    df_filtered = df.copy()
    if search:
        mask = (df_filtered['School'].str.contains(search, case=False, na=False) |
                df_filtered['Profile'].str.contains(search, case=False, na=False))
        df_filtered = df_filtered[mask]
    
    r1_rel = f'R1_{gender_label}_Reliable'
    if show_reliable and r1_rel in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[r1_rel] == True]
    
    # Results table
    r1_pred = f'R1_{gender_label}_Predicted'
    r1_conf = f'R1_{gender_label}_Confidence'
    r2_pred = f'R2_{gender_label}_Predicted'
    r2_conf = f'R2_{gender_label}_Confidence'
    
    display_cols = ['School', 'Profile']
    rename_map = {}
    for col, name in [(r1_pred, 'R1 Predicted'), (r1_conf, 'R1 Conf%'),
                      (r2_pred, 'R2 Predicted'), (r2_conf, 'R2 Conf%'), (r1_rel, 'Reliable')]:
        if col in df_filtered.columns:
            display_cols.append(col)
            rename_map[col] = name
    
    df_display = df_filtered[display_cols].sort_values(r1_pred, ascending=False).rename(columns=rename_map)
    st.dataframe(df_display, use_container_width=True, height=400)
    
    st.download_button("📥 Download CSV", df_filtered.to_csv(index=False).encode('utf-8'),
                       f"predictions_{predict_year}_{gender}.csv", "text/csv")
    
    # Historical trends
    with st.expander("📉 Historical Trends"):
        schools = sorted(df_filtered['School'].unique().tolist())
        selected_school = st.selectbox("Select school", options=[""] + schools)
        
        if selected_school:
            df_hist = load_historical_data(HISTORICAL_YEARS)
            school_hist = df_hist[df_hist['School'] == selected_school]
            target_col = f'R1_Min_{gender_label}'
            
            if not school_hist.empty and target_col in school_hist.columns:
                fig = go.Figure()
                for profile in school_hist['Profile'].unique():
                    pdata = school_hist[school_hist['Profile'] == profile].sort_values('Year')
                    if (pdata[target_col] > 0).any():
                        fig.add_trace(go.Scatter(x=pdata['Year'], y=pdata[target_col],
                                                 mode='lines+markers', name=profile[:35]))
                fig.update_layout(title=f"R1 History - {selected_school[:50]}", 
                                  xaxis_title="Year", yaxis_title="R1 Score")
                st.plotly_chart(fig, use_container_width=True)


def validation_page():
    st.header("✅ Model Validation")
    
    # Settings in columns at top
    col1, col2, col3 = st.columns(3)
    with col1:
        test_year = st.selectbox("Test Year", options=[2025, 2024, 2023])
    with col2:
        available_train = [y for y in HISTORICAL_YEARS if y < test_year]
        train_years = st.multiselect("Training Years", options=available_train, default=available_train)
    with col3:
        gender = st.selectbox("Gender", options=["female", "male"],
                              format_func=lambda x: "Жени (Female)" if x == "female" else "Мъже (Male)",
                              key="val_gender")
    
    if len(train_years) < 2:
        st.error("Select at least 2 training years")
        return
    
    gender_label = "Female" if gender == "female" else "Male"
    
    with st.spinner("Running validation..."):
        df, metrics = load_validation(train_years, test_year, gender)
    
    if df.empty:
        st.error("No validation results.")
        return
    
    # Metrics for R1 and R2
    st.subheader("📊 Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    r1_m = metrics.get(f"R1_{gender_label}", {})
    r2_m = metrics.get(f"R2_{gender_label}", {})
    
    with col1:
        st.metric("R1 MAE (All)", f"{r1_m.get('mae_existing', 0):.1f} pts")
    with col2:
        st.metric("R1 MAE (Reliable)", f"{r1_m.get('mae_reliable', 0):.1f} pts")
    with col3:
        st.metric("R2 MAE (All)", f"{r2_m.get('mae_existing', 0):.1f} pts")
    with col4:
        st.metric("R2 MAE (Reliable)", f"{r2_m.get('mae_reliable', 0):.1f} pts")
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search school or profile", "", key="val_search")
    with col2:
        show_reliable = st.checkbox("Reliable only", value=False, key="val_reliable")
    
    df_filtered = df.copy()
    if search:
        mask = (df_filtered['School'].str.contains(search, case=False, na=False) |
                df_filtered['Profile'].str.contains(search, case=False, na=False))
        df_filtered = df_filtered[mask]
    
    r1_rel = f'R1_{gender_label}_Reliable'
    if show_reliable and r1_rel in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[r1_rel] == True]
    
    # Results table with R1 and R2
    r1_actual = f'R1_{gender_label}_Actual'
    r1_pred = f'R1_{gender_label}_Predicted'
    r1_err = f'R1_{gender_label}_Abs_Error'
    r2_actual = f'R2_{gender_label}_Actual'
    r2_pred = f'R2_{gender_label}_Predicted'
    r2_err = f'R2_{gender_label}_Abs_Error'
    
    display_cols = ['School', 'Profile']
    rename_map = {}
    for col, name in [(r1_actual, 'R1 Actual'), (r1_pred, 'R1 Pred'), (r1_err, 'R1 Err'),
                      (r2_actual, 'R2 Actual'), (r2_pred, 'R2 Pred'), (r2_err, 'R2 Err'),
                      (r1_rel, 'Reliable')]:
        if col in df_filtered.columns:
            display_cols.append(col)
            rename_map[col] = name
    
    sort_col = r1_err if r1_err in df_filtered.columns else display_cols[0]
    df_display = df_filtered[display_cols].sort_values(sort_col, ascending=False).rename(columns=rename_map)
    
    st.subheader("📋 Results (sorted by R1 error)")
    st.dataframe(df_display, use_container_width=True, height=400)
    
    # Visualizations
    with st.expander("📈 Visualizations"):
        tab1, tab2 = st.tabs(["Predicted vs Actual", "Error Distribution"])
        
        with tab1:
            if r1_actual in df_filtered.columns and r1_pred in df_filtered.columns:
                fig = px.scatter(df_filtered, x=r1_actual, y=r1_pred,
                                 hover_data=['School', 'Profile'], title="R1: Predicted vs Actual")
                min_v = min(df_filtered[r1_actual].min(), df_filtered[r1_pred].min())
                max_v = max(df_filtered[r1_actual].max(), df_filtered[r1_pred].max())
                fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                         mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            err_col = f'R1_{gender_label}_Error'
            if err_col in df_filtered.columns:
                fig = px.histogram(df_filtered, x=err_col, nbins=40, title="R1 Error Distribution")
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
    
    st.download_button("📥 Download CSV", df_filtered.to_csv(index=False).encode('utf-8'),
                       f"validation_{test_year}_{gender}.csv", "text/csv")


if __name__ == "__main__":
    main()
