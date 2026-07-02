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
from nvo.services import run_predictions, run_validation, run_analysis
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
    
    # Use sidebar for navigation (more stable than tabs with text inputs)
    page = st.sidebar.radio("Navigate", ["🔮 Predictions", "✅ Validation", "📊 Analysis"], label_visibility="collapsed")
    
    if page == "🔮 Predictions":
        prediction_page()
    elif page == "✅ Validation":
        validation_page()
    else:
        analysis_page()
    
    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with ❤️ using XGBoost\n\n"
        "[GitHub](https://github.com/zhivko-georgiev/nvo7-predictor)"
    )


def prediction_page():
    st.header("🔮 Predictions")
    
    # Settings in columns at top instead of sidebar
    col1, col2, col3 = st.columns(3)
    with col1:
        train_years = st.multiselect(
            "Training Years",
            options=HISTORICAL_YEARS,
            default=[2022, 2023, 2024, 2025]
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
    
    r1_pred = f'R1_{gender_label}_Predicted'
    r2_pred = f'R2_{gender_label}_Predicted'
    r1_rel = f'R1_{gender_label}_Reliable'
    r1_conf = f'R1_{gender_label}_Confidence'
    r2_conf = f'R2_{gender_label}_Confidence'
    
    # Points filter for qualification check
    st.subheader("🎯 Check My Chances")
    col1, col2 = st.columns([1, 3])
    with col1:
        my_points = st.number_input("Enter your points", min_value=0.0, max_value=500.0, value=0.0, step=0.5, key="my_points")
    
    if my_points > 0:
        # Filter schools where user can qualify
        r1_qualify = df[df[r1_pred] <= my_points].copy() if r1_pred in df.columns else pd.DataFrame()
        r2_qualify = df[df[r2_pred] <= my_points].copy() if r2_pred in df.columns else pd.DataFrame()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**R1: {len(r1_qualify)} profiles** you may qualify for")
        with col2:
            st.markdown(f"**R2: {len(r2_qualify)} profiles** you may qualify for")
        
        tab_r1, tab_r2 = st.tabs(["Round 1 Options", "Round 2 Options"])
        
        with tab_r1:
            if not r1_qualify.empty:
                display_cols = ['School', 'Profile', r1_pred, r1_conf]
                display_cols = [c for c in display_cols if c in r1_qualify.columns]
                df_r1 = r1_qualify[display_cols].sort_values(r1_pred, ascending=False)
                df_r1 = df_r1.rename(columns={r1_pred: 'R1 Predicted', r1_conf: 'R1 Conf%'})
                st.dataframe(df_r1, width="stretch", hide_index=True)
            else:
                st.info("No profiles found for R1 with your points")
        
        with tab_r2:
            if not r2_qualify.empty:
                display_cols = ['School', 'Profile', r2_pred, r2_conf]
                display_cols = [c for c in display_cols if c in r2_qualify.columns]
                df_r2 = r2_qualify[display_cols].sort_values(r2_pred, ascending=False)
                df_r2 = df_r2.rename(columns={r2_pred: 'R2 Predicted', r2_conf: 'R2 Conf%'})
                st.dataframe(df_r2, width="stretch", hide_index=True)
            else:
                st.info("No profiles found for R2 with your points")
        
        st.markdown("---")
    
    # Metrics
    st.subheader("📋 All Predictions")
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
    
    if show_reliable and r1_rel in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[r1_rel] == True]
    
    # Results table
    display_cols = ['School', 'Profile']
    rename_map = {}
    for col, name in [(r1_pred, 'R1 Predicted'), (r1_conf, 'R1 Conf%'),
                      (r2_pred, 'R2 Predicted'), (r2_conf, 'R2 Conf%'), (r1_rel, 'Reliable')]:
        if col in df_filtered.columns:
            display_cols.append(col)
            rename_map[col] = name
    
    df_display = df_filtered[display_cols].sort_values(r1_pred, ascending=False).rename(columns=rename_map)
    st.dataframe(df_display, width="stretch", height=400)
    
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
                                  xaxis_title="Year", yaxis_title="R1 Score",
                                  xaxis=dict(dtick=1))
                st.plotly_chart(fig, width="stretch")


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
    st.dataframe(df_display, width="stretch", height=400)
    
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
                st.plotly_chart(fig, width="stretch")
        
        with tab2:
            err_col = f'R1_{gender_label}_Error'
            if err_col in df_filtered.columns:
                fig = px.histogram(df_filtered, x=err_col, nbins=40, title="R1 Error Distribution")
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, width="stretch")
    
    st.download_button("📥 Download CSV", df_filtered.to_csv(index=False).encode('utf-8'),
                       f"validation_{test_year}_{gender}.csv", "text/csv")


def build_trend_table(r1_scores, r2_scores, r3_scores):
    """Build a trend table from R1, R2, and R3 scores."""
    r1_by_year = {y: s for y, s in r1_scores}
    r2_by_year = {y: s for y, s in r2_scores}
    r3_by_year = {y: s for y, s in r3_scores}
    all_years = sorted(set(r1_by_year.keys()) | set(r2_by_year.keys()) | set(r3_by_year.keys()))
    
    rows = []
    prev_r1, prev_r2, prev_r3 = None, None, None
    for year in all_years:
        r1, r2, r3 = r1_by_year.get(year), r2_by_year.get(year), r3_by_year.get(year)
        row = {'Year': year}
        row['R1'] = f"{r1:.0f}" if r1 else "-"
        row['R1 Δ'] = f"{r1 - prev_r1:+.0f}" if r1 and prev_r1 else "-"
        row['R2'] = f"{r2:.0f}" if r2 else "-"
        row['R2 Δ'] = f"{r2 - prev_r2:+.0f}" if r2 and prev_r2 else "-"
        row['R3'] = f"{r3:.0f}" if r3 else "-"
        row['R3 Δ'] = f"{r3 - prev_r3:+.0f}" if r3 and prev_r3 else "-"
        row['R1→R2'] = f"{r2 - r1:+.0f}" if r1 and r2 else "-"
        row['R2→R3'] = f"{r3 - r2:+.0f}" if r2 and r3 else "-"
        row['R1→R3'] = f"{r3 - r1:+.0f}" if r1 and r3 else "-"
        rows.append(row)
        if r1: prev_r1 = r1
        if r2: prev_r2 = r2
        if r3: prev_r3 = r3
    return pd.DataFrame(rows)


def analysis_page():
    st.header("📊 Historical Analysis")
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        years = st.multiselect("Years to Analyze", options=HISTORICAL_YEARS, default=HISTORICAL_YEARS)
    with col2:
        gender = st.selectbox("Gender", options=["female", "male"],
                              format_func=lambda x: "Жени (Female)" if x == "female" else "Мъже (Male)",
                              key="analysis_gender")
    
    if len(years) < 1:
        st.error("Select at least 1 year")
        return
    
    gender_label = "Female" if gender == "female" else "Male"
    
    # School filter
    search = st.text_input("🔍 Filter by school name", "", key="analysis_search")
    school_filter = [search] if search else None
    
    with st.spinner("Analyzing data..."):
        yearly_stats, trends = run_analysis(years, cfg.data['files_dir'], gender, school_filter)
    
    # Yearly aggregate statistics
    st.subheader("📈 Yearly Statistics")
    
    stats_data = []
    for year, stats in sorted(yearly_stats.items()):
        row = {'Year': year, 'Records': stats['records'], 'Schools': stats['schools']}
        for rnd, rnd_stats in stats['rounds'].items():
            row[f'R{rnd} Min'] = rnd_stats['min']
            row[f'R{rnd} Max'] = rnd_stats['max']
            row[f'R{rnd} Mean'] = round(rnd_stats['mean'], 1)
        stats_data.append(row)
    
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), width="stretch")
    
    # Visualize yearly trends
    if len(years) > 1 and stats_data:
        df_stats = pd.DataFrame(stats_data)
        fig = go.Figure()
        for col in ['R1 Mean', 'R2 Mean']:
            if col in df_stats.columns:
                fig.add_trace(go.Scatter(x=df_stats['Year'], y=df_stats[col],
                                         mode='lines+markers', name=col))
        fig.update_layout(title=f"Average Cutoff Scores by Year ({gender_label})",
                          xaxis_title="Year", yaxis_title="Score",
                          xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")
    
    # School trends
    if school_filter and trends:
        st.subheader(f"📉 Trends - Cutoff Scores ({gender_label})")
        
        current_school = None
        for (school, profile), data in sorted(trends.items()):
            r1_scores = data['R1']
            r2_scores = data['R2']
            r3_scores = data.get('R3', [])
            
            if len(r1_scores) < 2 and len(r2_scores) < 2 and len(r3_scores) < 2:
                continue
            
            if school != current_school:
                st.markdown(f"**{school}**")
                current_school = school
            
            with st.expander(f"  {profile[:60]}"):
                # Trend table
                df_trend = build_trend_table(r1_scores, r2_scores, r3_scores)
                st.dataframe(df_trend, width="stretch", hide_index=True)
                
                # Chart below table
                fig = go.Figure()
                if r1_scores:
                    fig.add_trace(go.Scatter(x=[y for y, _ in r1_scores], y=[s for _, s in r1_scores],
                                             mode='lines+markers', name='R1'))
                if r2_scores:
                    fig.add_trace(go.Scatter(x=[y for y, _ in r2_scores], y=[s for _, s in r2_scores],
                                             mode='lines+markers', name='R2'))
                if r3_scores:
                    fig.add_trace(go.Scatter(x=[y for y, _ in r3_scores], y=[s for _, s in r3_scores],
                                             mode='lines+markers', name='R3'))
                fig.update_layout(xaxis_title="Year", yaxis_title="Score", height=250, margin=dict(t=10),
                                  xaxis=dict(dtick=1))
                st.plotly_chart(fig, width="stretch")
    
    elif not school_filter:
        st.info("💡 Enter a school name above to see detailed trends with year-over-year changes")


if __name__ == "__main__":
    main()
