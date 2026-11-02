"""
Student Academic Performance Analytics – Streamlit dashboard
Big Data Analytics case study (descriptive → predictive → prescriptive).

Run locally:  streamlit run app.py
"""
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A

# ----------------------------------------------------------------------------
# Page setup & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAM = ["U. Sampath", "P. Kartheek", "T. Abhishek"]  # edit or empty the list as you like
RISK_COLORS = {"High": "#DC2626", "Medium": "#F59E0B", "Low": "#16A34A"}
PALETTE = ["#4F46E5", "#06B6D4", "#F59E0B", "#EC4899", "#10B981", "#8B5CF6", "#EF4444"]
CHART_FONT = dict(family="sans-serif", size=13)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {background: linear-gradient(120deg,#1e3a8a 0%,#4f46e5 55%,#06b6d4 100%);
           padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.1rem;
           box-shadow: 0 6px 18px rgba(79,70,229,.25);}
    .hero h1 {margin: 0; font-size: 2rem; color: #ffffff !important;}
    .hero p  {margin: .35rem 0 0; color: #e0e7ff !important; font-size: 1.02rem;}
    .hero .tag {display:inline-block; background: rgba(255,255,255,.18); color:#fff;
                padding: .15rem .7rem; border-radius: 999px; font-size:.75rem;
                letter-spacing:.06em; margin-bottom:.5rem;}
    .kpi {background:#ffffff; border:1px solid #e5e7eb; border-left:5px solid var(--accent,#4f46e5);
          border-radius:12px; padding:.85rem 1rem; box-shadow:0 1px 3px rgba(0,0,0,.06); min-height:105px;}
    .kpi .label {font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:#6b7280;}
    .kpi .value {font-size:1.75rem; font-weight:700; color:#111827; line-height:1.25;}
    .kpi .sub   {font-size:.78rem; color:#6b7280;}
    .insight {background:#eef2ff; border-left:4px solid #4f46e5; border-radius:8px;
              padding:.8rem 1rem; color:#1e1b4b;}
    .footer {text-align:center; color:#6b7280; font-size:.8rem; margin-top:2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub="", accent="#4F46E5"):
    col.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def fmt(x, suffix="", nd=1):
    return "–" if x is None or pd.isna(x) else f"{x:.{nd}f}{suffix}"


def style(fig, height=380):
    fig.update_layout(
        template="plotly_white", height=height, font=CHART_FONT,
        margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
        colorway=PALETTE,
    )
    return fig


def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------------------------------------------------------
# Cached data steps
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return A.generate_sample_data()


@st.cache_data(show_spinner=False)
def read_upload(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def run_cleaning(raw: pd.DataFrame):
    return A.clean_data(raw)


@st.cache_data(show_spinner="Training models…")
def run_ml(data: pd.DataFrame, pass_mark: float):
    return A.train_risk_models(data, pass_mark=pass_mark)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="tag">BIG DATA ANALYTICS · CASE STUDY</div>
      <h1>🎓 Student Academic Performance Analytics</h1>
      <p>Turning student records into insights, early warnings and data-driven academic decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar: data source + settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Control panel")
    source = st.radio("Data source", ["Built-in sample data", "Upload my CSV"])
    uploaded = None
    if source == "Upload my CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        st.caption("Required columns: " + ", ".join(A.REQUIRED_COLUMNS))
    st.download_button(
        "⬇️ Download CSV template",
        data=to_csv(load_sample()),
        file_name="student_data_template.csv",
        mime="text/csv",
        key="dl_template",
    )
    st.warning(
        "Use synthetic or institution-approved data only. Avoid uploading real, "
        "identifiable student records to a public app.",
        icon="🔒",
    )

    with st.expander("⚙️ Marks & risk thresholds", expanded=False):
        pass_mark = st.slider("Pass mark (average marks)", 30, 60, 40)
        att_thr = st.slider("Low attendance below (%)", 50, 90, 75)
        marks_thr = st.slider("Low marks below", 40, 70, 50)
        asg_thr = st.slider("Low assignment completion below (%)", 40, 90, 70)

# ----------------------------------------------------------------------------
# Load → clean
# ----------------------------------------------------------------------------
if source == "Upload my CSV":
    if uploaded is None:
        st.info("👈 Upload a CSV file in the sidebar, or switch to the built-in sample data. "
                "You can download the template to see the expected format.")
        st.stop()
    try:
        raw = read_upload(uploaded.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that file as CSV: {exc}")
        st.stop()
else:
    raw = load_sample()

try:
    clean, report = run_cleaning(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if clean.empty:
    st.error("No usable rows remain after cleaning. Check the student_id, course and semester columns.")
    st.stop()

# ----------------------------------------------------------------------------
# Metrics, risk, ML
# ----------------------------------------------------------------------------
full = A.derive_metrics(clean, pass_mark)
full = A.compute_risk(full, att_thr, marks_thr, asg_thr, pass_mark)
ml = run_ml(full[A.FEATURES + ["assessment2"]], float(pass_mark))
if ml["ok"]:
    full["ml_risk_prob"] = ml["prob"]

# ----------------------------------------------------------------------------
# Sidebar filters (need the data)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔎 Filters")
    courses = sorted(full["course"].unique())
    semesters = sorted(full["semester"].unique())
    sel_courses = st.multiselect("Course", courses, default=courses)
    sel_sems = st.multiselect("Semester", semesters, default=semesters)
    att_range = st.slider("Attendance range (%)", 0, 100, (0, 100))
    st.divider()
    if TEAM:
        st.caption("Team: " + " · ".join(TEAM))

mask = full["course"].isin(sel_courses) & full["semester"].isin(sel_sems)
mask &= full["attendance_pct"].between(*att_range) | full["attendance_pct"].isna()
df = full[mask]
if df.empty:
    st.warning("No records match the current filters.")
    st.stop()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_dash, tab_att, tab_course, tab_risk, tab_ml, tab_data = st.tabs(
    ["📊 Dashboard", "📈 Attendance & Marks", "📚 Course analysis",
     "🚨 Risk identification", "🤖 ML prediction", "🗂️ Data & downloads"]
)

# ============================== DASHBOARD ====================================
with tab_dash:
    c = st.columns(5)
    kpi(c[0], "Students", f"{df['student_id'].nunique():,}", f"{len(df):,} course records")
    kpi(c[1], "Avg attendance", fmt(df["attendance_pct"].mean(), "%"), "across selected records", "#06B6D4")
    kpi(c[2], "Avg marks", fmt(df["average_marks"].mean()), f"pass mark = {pass_mark}", "#8B5CF6")
    kpi(c[3], "Pass rate", fmt(A.pass_rate(df), "%"), "marks ≥ pass mark", "#16A34A")
    high = int((df["risk_level"] == "High").sum())
    kpi(c[4], "High-risk records", f"{high:,}", f"{high / len(df) * 100:.1f}% of records", "#DC2626")
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        band = (df["performance_band"].value_counts().reindex(A.PERFORMANCE_LABELS)
                .fillna(0).rename_axis("band").reset_index(name="records"))
        fig = px.bar(band, x="band", y="records", text="records",
                     title="Performance distribution", color="band",
                     color_discrete_sequence=["#DC2626", "#F59E0B", "#FACC15", "#22C55E", "#15803D"])
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records")
        fig.update_traces(textposition="outside")
        st.plotly_chart(style(fig))
    with right:
        rk = df["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        rk.columns = ["risk", "records"]
        fig = px.pie(rk, names="risk", values="records", hole=0.55, color="risk",
                     color_discrete_map=RISK_COLORS, title="Risk level mix")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig))

    left, right = st.columns(2)
    with left:
        trend = (df.groupby(["semester", "course"], as_index=False)["average_marks"].mean().round(2))
        fig = px.line(trend, x="semester", y="average_marks", color="course", markers=True,
                      title="Average marks by semester")
        fig.update_layout(xaxis_title="Semester", yaxis_title="Average marks")
        st.plotly_chart(style(fig))
    with right:
        att_band = (df.groupby("attendance_band", observed=False)
                    .agg(avg_marks=("average_marks", "mean")).reindex(A.ATTENDANCE_LABELS).reset_index())
        fig = px.bar(att_band, x="attendance_band", y="avg_marks", text_auto=".1f",
                     title="Average marks by attendance band", color_discrete_sequence=["#4F46E5"])
        fig.update_layout(xaxis_title="Attendance band", yaxis_title="Average marks")
        st.plotly_chart(style(fig))

    # Auto-generated insights
    ck = A.course_kpis(df)
    corr = df[["attendance_pct", "average_marks"]].corr().iloc[0, 1]
    best, worst = ck.sort_values("avg_marks").iloc[-1], ck.sort_values("avg_marks").iloc[0]
    bullets = []
    if pd.notna(corr):
        strength = "strong" if abs(corr) >= 0.6 else "moderate" if abs(corr) >= 0.3 else "weak"
        bullets.append(f"Attendance and average marks show a **{strength} {'positive' if corr > 0 else 'negative'}** "
                       f"relationship (r = {corr:.2f}).")
    if len(ck) > 1:
        bullets.append(f"**{best['course']}** has the highest average marks ({best['avg_marks']:.1f}); "
                       f"**{worst['course']}** the lowest ({worst['avg_marks']:.1f}).")
    bullets.append(f"**{high}** record(s) are rated *High* risk and may benefit from supportive review.")
    st.markdown("#### 💡 Key insights")
    html = "<br>".join("• " + b for b in bullets)
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
    html = re.sub(r"\*(.+?)\*", r"<i>\1</i>", html)
    st.markdown('<div class="insight">' + html + "</div>", unsafe_allow_html=True)

# ========================= ATTENDANCE & MARKS ================================
with tab_att:
    st.subheader("How do attendance and marks relate?")
    sc = df.dropna(subset=["attendance_pct", "average_marks"])
    left, right = st.columns(2)
    with left:
        fig = px.scatter(sc, x="attendance_pct", y="average_marks", color="course", opacity=0.75,
                         hover_data=["student_id", "semester"], title="Attendance vs average marks")
        if len(sc) >= 3 and np.ptp(sc["attendance_pct"]) > 0:
            m, b = np.polyfit(sc["attendance_pct"], sc["average_marks"], 1)
            xs = np.array([sc["attendance_pct"].min(), sc["attendance_pct"].max()])
            fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                     line=dict(color="#111827", dash="dash")))
        fig.add_hline(y=pass_mark, line_dash="dot", line_color="#DC2626",
                      annotation_text="Pass mark", annotation_position="bottom right")
        fig.update_layout(xaxis_title="Attendance (%)", yaxis_title="Average marks")
        st.plotly_chart(style(fig, 420))
    with right:
        fig = px.box(sc.dropna(subset=["attendance_band"]), x="attendance_band", y="average_marks",
                     category_orders={"attendance_band": A.ATTENDANCE_LABELS}, color="attendance_band",
                     color_discrete_sequence=["#DC2626", "#F59E0B", "#22C55E", "#15803D"],
                     title="Marks spread by attendance band", points="all")
        fig.update_layout(showlegend=False, xaxis_title="Attendance band", yaxis_title="Average marks")
        st.plotly_chart(style(fig, 420))

    left, right = st.columns(2)
    with left:
        cols = {"attendance_pct": "Attendance", "assessment1": "Assessment 1",
                "assessment2": "Assessment 2", "assignment_completion_pct": "Assignments",
                "average_marks": "Avg marks"}
        corr_m = df[list(cols)].rename(columns=cols).corr()
        fig = px.imshow(corr_m, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation heatmap", aspect="auto")
        st.plotly_chart(style(fig, 400))
    with right:
        imp = df.dropna(subset=["improvement"])
        fig = px.histogram(imp, x="improvement", nbins=30, title="Change from Assessment 1 → 2",
                           color_discrete_sequence=["#06B6D4"])
        fig.add_vline(x=0, line_dash="dash", line_color="#111827")
        fig.update_layout(xaxis_title="Marks gained (Assessment 2 − Assessment 1)", yaxis_title="Records")
        st.plotly_chart(style(fig, 400))
        if len(imp):
            st.caption(f"{(imp['improvement'] > 0).mean() * 100:.0f}% of records improved; "
                       f"average change {imp['improvement'].mean():+.1f} marks.")

    st.markdown("##### Summary by attendance band")
    summary = (df.groupby("attendance_band", observed=False)
               .agg(records=("student_id", "size"), avg_marks=("average_marks", "mean"),
                    avg_assignment=("assignment_completion_pct", "mean"))
               .round(1).reindex(A.ATTENDANCE_LABELS).reset_index())
    pr = (df[df["pass_status"] != "Unknown"].groupby("attendance_band", observed=False)["pass_status"]
          .apply(lambda s: round((s == "Pass").mean() * 100, 1)).rename("pass_rate_%").reset_index())
    st.dataframe(summary.merge(pr, on="attendance_band", how="left"), hide_index=True)

# ============================ COURSE ANALYSIS ================================
with tab_course:
    st.subheader("Course-wise performance")
    ck = A.course_kpis(df)
    left, right = st.columns(2)
    with left:
        long = ck.melt(id_vars="course", value_vars=["avg_attendance", "avg_marks", "avg_assignment"],
                       var_name="metric", value_name="value")
        long["metric"] = long["metric"].map({"avg_attendance": "Attendance %", "avg_marks": "Avg marks",
                                             "avg_assignment": "Assignment %"})
        fig = px.bar(long, x="course", y="value", color="metric", barmode="group", text_auto=".0f",
                     title="Attendance, marks and assignments by course")
        fig.update_layout(xaxis_title="", yaxis_title="Value")
        st.plotly_chart(style(fig, 420))
    with right:
        fig = px.bar(ck.sort_values("pass_rate"), x="pass_rate", y="course", orientation="h",
                     text_auto=".1f", title="Pass rate by course (%)", color="pass_rate",
                     color_continuous_scale="RdYlGn", range_color=[50, 100])
        fig.update_layout(xaxis_title="Pass rate (%)", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(style(fig, 420))

    left, right = st.columns(2)
    with left:
        fig = px.box(df.dropna(subset=["average_marks"]), x="course", y="average_marks", color="course",
                     points="outliers", title="Marks distribution by course")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Average marks")
        st.plotly_chart(style(fig, 400))
    with right:
        pv = df.pivot_table(index="course", columns="semester", values="average_marks", aggfunc="mean")
        fig = px.imshow(pv.round(1), text_auto=".1f", color_continuous_scale="Blues",
                        title="Average marks: course × semester", aspect="auto")
        fig.update_layout(xaxis_title="Semester", yaxis_title="")
        st.plotly_chart(style(fig, 400))

    st.markdown("##### Course KPI table")
    st.dataframe(
        ck.rename(columns={"avg_attendance": "avg attendance %", "avg_marks": "avg marks",
                           "avg_assignment": "avg assignment %", "high_risk": "high-risk records",
                           "pass_rate": "pass rate %"}),
        hide_index=True,
    )
    st.download_button("⬇️ Download course KPIs (CSV)", to_csv(ck), "course_kpis.csv", "text/csv",
                       key="dl_course")

# ============================ RISK IDENTIFICATION ============================
with tab_risk:
    st.subheader("Students who may benefit from supportive review")
    st.caption("Rule-based flags are transparent and adjustable (sidebar → thresholds). They are meant to "
               "prioritise mentoring/tutoring conversations — **not** to make disciplinary decisions.")
    c = st.columns(3)
    for col, lvl in zip(c, ["High", "Medium", "Low"]):
        kpi(col, f"{lvl} risk", f"{int((df['risk_level'] == lvl).sum()):,}", "records", RISK_COLORS[lvl])
    st.write("")

    left, right = st.columns(2)
    with left:
        sc = df.dropna(subset=["attendance_pct", "average_marks"])
        fig = px.scatter(sc, x="attendance_pct", y="average_marks", color="risk_level",
                         color_discrete_map=RISK_COLORS, category_orders={"risk_level": ["High", "Medium", "Low"]},
                         hover_data=["student_id", "course", "risk_reasons"], opacity=0.8,
                         title="Risk map: attendance vs marks")
        fig.add_vline(x=att_thr, line_dash="dot", line_color="#6B7280")
        fig.add_hline(y=marks_thr, line_dash="dot", line_color="#6B7280")
        fig.update_layout(xaxis_title="Attendance (%)", yaxis_title="Average marks")
        st.plotly_chart(style(fig, 400))
    with right:
        rc = df.groupby(["course", "risk_level"]).size().reset_index(name="records")
        fig = px.bar(rc, x="course", y="records", color="risk_level", color_discrete_map=RISK_COLORS,
                     category_orders={"risk_level": ["High", "Medium", "Low"]},
                     title="Risk level by course", barmode="stack")
        fig.update_layout(xaxis_title="", yaxis_title="Records")
        st.plotly_chart(style(fig, 400))

    levels = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
    view = A.sort_by_priority(df[df["risk_level"].isin(levels)]).copy()
    show_cols = ["student_id", "course", "semester", "attendance_pct", "average_marks",
                 "assignment_completion_pct", "risk_level", "risk_reasons", "suggested_action"]
    col_cfg = {
        "attendance_pct": st.column_config.ProgressColumn("Attendance %", min_value=0, max_value=100, format="%.0f"),
        "average_marks": st.column_config.ProgressColumn("Avg marks", min_value=0, max_value=100, format="%.1f"),
        "assignment_completion_pct": st.column_config.ProgressColumn("Assignments %", min_value=0, max_value=100, format="%.0f"),
    }
    if "ml_risk_prob" in view:
        view["ml_risk_%"] = (view["ml_risk_prob"] * 100).round(0)
        show_cols.insert(7, "ml_risk_%")
        col_cfg["ml_risk_%"] = st.column_config.ProgressColumn("ML risk %", min_value=0, max_value=100, format="%.0f")
    st.dataframe(view[show_cols], hide_index=True, column_config=col_cfg)
    st.download_button("⬇️ Download flagged students (CSV)", to_csv(view[show_cols]),
                       "students_for_supportive_review.csv", "text/csv", key="dl_risk")

    st.markdown("##### 🔍 Student drill-down")
    sid = st.selectbox("Select a student", sorted(df["student_id"].unique()))
    rec = df[df["student_id"] == sid]
    st.dataframe(rec[["course", "semester", "attendance_pct", "assessment1", "assessment2",
                      "assignment_completion_pct", "average_marks", "risk_level"]], hide_index=True)
    for _, r in rec.iterrows():
        st.markdown(f"**{r['course']} · {r['semester']}** — {r['risk_level']} risk. "
                    f"Reasons: {r['risk_reasons']}. Suggested action: {r['suggested_action']}.")

# ============================== ML PREDICTION ================================
with tab_ml:
    st.subheader("Early-warning model (predictive analytics)")
    st.markdown(
        "The model predicts whether a student is **likely to fail Assessment 2** "
        f"(score < {pass_mark}) using only **attendance, Assessment 1 score and assignment completion**. "
        "Assessment 2 is *not* an input, so there is no data leakage. Two models are compared and the "
        "better one (by F1 on the at-risk class) is used below."
    )
    if not ml["ok"]:
        st.warning(ml["message"])
    else:
        c = st.columns(4)
        kpi(c[0], "Best model", ml["best_name"], "by F1 (at-risk)", "#4F46E5")
        best_row = ml["metrics"].set_index("Model").loc[ml["best_name"]]
        kpi(c[1], "Recall (at-risk)", f"{best_row['Recall (at-risk)']:.0%}", "at-risk students caught", "#16A34A")
        kpi(c[2], "ROC AUC", f"{best_row['ROC AUC']:.2f}", "1.0 = perfect, 0.5 = random", "#06B6D4")
        kpi(c[3], "Baseline accuracy", f"{ml['baseline_accuracy']:.0%}", "always predict 'passes'", "#F59E0B")
        st.write("")
        st.caption(f"Trained on {ml['n_train']} records, tested on {ml['n_test']} held-out records "
                   f"({ml['at_risk_rate']:.0%} of labelled records are at-risk). Accuracy alone can be misleading "
                   "when few students fail — look at recall and precision for the at-risk class.")
        st.dataframe(ml["metrics"], hide_index=True)

        left, right = st.columns(2)
        with left:
            cm = ml["confusion"][ml["best_name"]]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Predicted OK", "Predicted at-risk"], y=["Actually OK", "Actually at-risk"],
                            title=f"Confusion matrix – {ml['best_name']} (test set)")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style(fig, 360))
        with right:
            imp = ml["importance"][ml["best_name"]].rename(index={
                "attendance_pct": "Attendance", "assessment1": "Assessment 1",
                "assignment_completion_pct": "Assignments"}).sort_values().reset_index()
            imp.columns = ["feature", "importance"]
            fig = px.bar(imp, x="importance", y="feature", orientation="h", text_auto=".2f",
                         title="What drives the prediction?", color_discrete_sequence=["#8B5CF6"])
            fig.update_layout(xaxis_title="Relative importance", yaxis_title="")
            st.plotly_chart(style(fig, 360))

        st.markdown("##### 🎯 What-if predictor")
        w1, w2, w3, w4 = st.columns([1, 1, 1, 1.2])
        a = w1.slider("Attendance %", 0, 100, 70)
        s1 = w2.slider("Assessment 1 score", 0, 100, 50)
        g = w3.slider("Assignment completion %", 0, 100, 70)
        p = float(ml["final_model"].predict_proba(
            pd.DataFrame([[a, s1, g]], columns=A.FEATURES))[0, 1])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Probability of failing Assessment 2"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4F46E5"},
                   "steps": [{"range": [0, 30], "color": "#DCFCE7"},
                             {"range": [30, 60], "color": "#FEF3C7"},
                             {"range": [60, 100], "color": "#FEE2E2"}]}))
        w4.plotly_chart(style(gauge, 230))

        st.markdown("##### ⚠️ Early warnings: highest predicted risk in current selection")
        top = df.dropna(subset=["ml_risk_prob"]).sort_values("ml_risk_prob", ascending=False).head(15).copy()
        top["ml_risk_%"] = (top["ml_risk_prob"] * 100).round(0)
        st.dataframe(
            top[["student_id", "course", "semester", "attendance_pct", "assessment1",
                 "assignment_completion_pct", "ml_risk_%", "risk_level"]],
            hide_index=True,
            column_config={"ml_risk_%": st.column_config.ProgressColumn(
                "ML risk %", min_value=0, max_value=100, format="%.0f")},
        )
        st.caption("Risk % for each record is an out-of-fold estimate (the record was not used to train the "
                   "model that scored it). Treat predictions as prompts for human review, not verdicts; "
                   "check fairness and validate on real data before any real use.")

# ============================= DATA & DOWNLOADS ==============================
with tab_data:
    st.subheader("Data quality report")
    c = st.columns(4)
    kpi(c[0], "Rows loaded", f"{report['rows_in']:,}", "", "#4F46E5")
    kpi(c[1], "Duplicates removed", f"{report['duplicates_removed']:,}", "same student + course + semester", "#F59E0B")
    kpi(c[2], "Invalid values blanked", f"{report['invalid_values_blanked']:,}", "outside 0–100", "#DC2626")
    kpi(c[3], "Rows after cleaning", f"{report['rows_out']:,}", f"{report['dropped_missing_keys']} dropped (no ID/course/semester)", "#16A34A")
    st.write("")
    if report["renamed_columns"]:
        st.caption("Columns renamed automatically: " +
                   ", ".join(f"{k} → {v}" for k, v in report["renamed_columns"].items()))
    miss = pd.DataFrame({"column": list(report["missing_by_column"]),
                         "missing values": list(report["missing_by_column"].values())})
    st.dataframe(miss, hide_index=True)
    st.caption("Missing numeric values stay blank in charts and are median-imputed only inside the ML pipeline.")

    st.subheader("Results table")
    out_cols = ["student_id", "course", "semester", "attendance_pct", "assessment1", "assessment2",
                "assignment_completion_pct", "average_marks", "pass_status", "attendance_band",
                "performance_band", "risk_level", "risk_reasons", "suggested_action"]
    if "ml_risk_prob" in df:
        out_cols.append("ml_risk_prob")
    results = A.sort_by_priority(df)[out_cols]
    st.dataframe(results, hide_index=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Full results (CSV)", to_csv(results), "student_academic_analytics_results.csv",
                       "text/csv", key="dl_full")
    flagged = results[results["risk_level"] != "Low"]
    d2.download_button("⬇️ Flagged students (CSV)", to_csv(flagged), "students_for_supportive_review.csv",
                       "text/csv", key="dl_flagged")
    d3.download_button("⬇️ Cleaned input data (CSV)", to_csv(clean), "cleaned_student_data.csv",
                       "text/csv", key="dl_clean")

st.markdown(
    '<div class="footer">Demonstration with synthetic data · Built with Streamlit, Pandas, Plotly &amp; scikit-learn</div>',
    unsafe_allow_html=True,
)
