from __future__ import annotations

from datetime import date
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# Page setup and theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NEST | India Property Investment Studio",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "india_housing_prices.csv"
HERO_PATH = APP_DIR / "assets" / "hero.svg"
TODAY_YEAR = date.today().year

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --navy:#102A43; --ink:#172B4D; --teal:#16B8A6; --coral:#FF8066; --sky:#5C8DFF; --paper:#F5F8FC; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#F5F8FC 0%,#F7FAFD 46%,#FFFFFF 100%); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#102A43 0%,#153B58 100%); }
    [data-testid="stSidebar"] * { color:#F3FAFF !important; }
    [data-testid="stSidebar"] input { color:#172B4D !important; }
    h1,h2,h3 { color:#102A43; font-family:'Manrope','DM Sans',sans-serif; letter-spacing:-.035em; }
    .block-container { padding-top:1.1rem; padding-bottom:3rem; max-width:1440px; }
    .hero-wrap { position:relative; overflow:hidden; border-radius:26px; min-height:280px; padding:36px 42px; margin:0 0 22px; color:white; background:#102A43; box-shadow:0 18px 46px rgba(16,42,67,.18); }
    .hero-img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; opacity:.92; }
    .hero-shade { position:absolute; inset:0; background:linear-gradient(90deg,rgba(7,24,47,.94) 0%,rgba(7,24,47,.78) 40%,rgba(7,24,47,.10) 100%); }
    .hero-content { position:relative; z-index:2; max-width:690px; }
    .eyebrow { color:#85E5D3; text-transform:uppercase; letter-spacing:.19em; font-size:.72rem; font-weight:800; }
    .hero-title { color:white !important; font-size:clamp(2rem,4.4vw,3.65rem); line-height:1.02; margin:.45rem 0 .7rem; font-weight:800; }
    .hero-copy { color:#D9E9F5; font-size:1.03rem; max-width:610px; line-height:1.6; }
    .hero-chip { display:inline-block; margin-top:13px; margin-right:8px; border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.09); color:#F5FCFF; border-radius:999px; padding:7px 12px; font-size:.78rem; }
    .section-kicker { color:#168E86; text-transform:uppercase; font-weight:800; letter-spacing:.14em; font-size:.72rem; margin-bottom:.25rem; }
    .soft-note { background:#EAF8F6; border:1px solid #C7EEE8; border-radius:14px; padding:12px 15px; color:#225B5A; font-size:.89rem; }
    .muted { color:#61758A; font-size:.88rem; }
    div[data-testid="stMetric"] { background:white; border:1px solid #E5ECF3; padding:17px 19px; border-radius:18px; box-shadow:0 7px 20px rgba(16,42,67,.045); }
    div[data-testid="stMetricLabel"] { color:#63778A; font-weight:700; }
    div[data-testid="stMetricValue"] { color:#102A43; font-family:'Manrope',sans-serif; }
    button[kind="primary"] { background:linear-gradient(100deg,#12A899,#1D75C9); border:0; border-radius:12px; font-weight:800; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; background:#EAF0F6; border-radius:16px; padding:7px; }
    .stTabs [data-baseweb="tab"] { background:transparent; border-radius:11px; padding:10px 17px; color:#52677D; font-weight:700; }
    .stTabs [aria-selected="true"] { background:white !important; color:#102A43 !important; box-shadow:0 3px 10px rgba(16,42,67,.09); }
    div[data-testid="stForm"] { background:white; border:1px solid #E5ECF3; border-radius:18px; padding:18px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Dataset loading and preparation
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Year_Built", "Furnished_Status", "Floor_No",
    "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status",
]
NUMERIC_COLUMNS = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
]
MODEL_NUMERIC = [
    "BHK", "Size_in_SqFt", "Year_Built", "Age_of_Property", "Floor_No",
    "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
]
MODEL_CATEGORICAL = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status", "Facing",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Owner_Type", "Amenities",
]
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL
PRICE_LABELS = ["Value", "Mid-market", "Premium"]
PALETTE = ["#16B8A6", "#5C8DFF", "#FF8066", "#A879E8", "#F4B942", "#3CBF83", "#EC6C9B"]


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("The CSV is missing required columns: " + ", ".join(missing))
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["State", "City", "Locality", "Property_Type", "Size_in_SqFt", "Price_in_Lakhs"])
    result = result.loc[(result["Size_in_SqFt"] > 0) & (result["Price_in_Lakhs"] >= 0)].copy()
    result["Calculated_Price_per_SqFt_Rs"] = result["Price_in_Lakhs"] * 100_000 / result["Size_in_SqFt"]
    if "ID" not in result:
        result["ID"] = range(1, len(result) + 1)
    if "_is_user_added" not in result.columns:
        result["_is_user_added"] = False
    else:
        result["_is_user_added"] = result["_is_user_added"].fillna(False).astype(bool)
    return result


@st.cache_data(show_spinner="Loading the property dataset…")
def load_bundled_data(path: str, file_size: int, modified_ns: int) -> pd.DataFrame:
    del file_size, modified_ns  # values are included in Streamlit's cache key
    return prepare_data(pd.read_csv(path, low_memory=False))


@st.cache_data(show_spinner="Reading uploaded data…")
def load_uploaded_data(raw_csv: bytes) -> pd.DataFrame:
    return prepare_data(pd.read_csv(BytesIO(raw_csv), low_memory=False))


if "new_listings" not in st.session_state:
    st.session_state.new_listings = pd.DataFrame(columns=REQUIRED_COLUMNS + ["ID", "_is_user_added"])
if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None

with st.sidebar:
    st.markdown("### 🏙️ NEST Studio")
    st.caption("Property intelligence for your next move")
    upload = st.file_uploader("Use an updated CSV dataset", type=["csv"], help="The uploaded CSV must keep the same column names as the supplied dataset.")

try:
    if upload is not None:
        upload_bytes = upload.getvalue()
        source_df = load_uploaded_data(upload_bytes)
        source_signature = sha256(upload_bytes).hexdigest()
        source_name = f"Uploaded · {upload.name}"
    else:
        stat = DATA_PATH.stat()
        source_df = load_bundled_data(str(DATA_PATH), stat.st_size, stat.st_mtime_ns)
        source_signature = f"{stat.st_size}-{stat.st_mtime_ns}"
        source_name = "Bundled India housing dataset"
except FileNotFoundError:
    st.error("Dataset file is missing. Place `india_housing_prices.csv` next to `app.py` in the GitHub repository.")
    st.stop()
except (OSError, pd.errors.ParserError, ValueError) as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if source_df.empty:
    st.error("This dataset has no usable property records.")
    st.stop()

extra_df = st.session_state.new_listings.copy()
if not extra_df.empty:
    extra_df = prepare_data(extra_df)
    extra_df["_is_user_added"] = True
    analysis_all = pd.concat([source_df, extra_df], ignore_index=True, sort=False)
else:
    analysis_all = source_df.copy()
    analysis_all["_is_user_added"] = False

with st.sidebar:
    st.markdown("---")
    st.markdown("#### Explore a market")
    state_options = sorted(analysis_all["State"].dropna().astype(str).unique().tolist())
    selected_states = st.multiselect("States", state_options, placeholder="All states")
    state_scope = analysis_all.loc[analysis_all["State"].astype(str).isin(selected_states)] if selected_states else analysis_all
    city_options = sorted(state_scope["City"].dropna().astype(str).unique().tolist())
    selected_cities = st.multiselect("Cities", city_options, placeholder="All cities")
    type_options = sorted(analysis_all["Property_Type"].dropna().astype(str).unique().tolist())
    selected_types = st.multiselect("Property types", type_options, placeholder="All property types")
    st.caption(f"Data source: {source_name}")
    st.caption("Added listings are included immediately in this session's charts and model training.")

filtered = analysis_all
if selected_states:
    filtered = filtered.loc[filtered["State"].astype(str).isin(selected_states)]
if selected_cities:
    filtered = filtered.loc[filtered["City"].astype(str).isin(selected_cities)]
if selected_types:
    filtered = filtered.loc[filtered["Property_Type"].astype(str).isin(selected_types)]
if filtered.empty:
    st.warning("No properties match those filters. Clear or change your selections in the sidebar.")
    st.stop()


# -----------------------------------------------------------------------------
# ML training: price regression + price-segment classification
# -----------------------------------------------------------------------------
def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=8, max_categories=50, sparse_output=True)),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, MODEL_NUMERIC), ("categorical", categorical_pipe, MODEL_CATEGORICAL)],
        remainder="drop",
        sparse_threshold=0.4,
    )


@st.cache_resource(show_spinner="Training the property models on the current dataset…")
def train_models(_data: pd.DataFrame, fingerprint: str) -> dict:
    del fingerprint  # included in the cache key; data is deliberately underscore-prefixed
    working = _data.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt"]).copy()
    working = working.loc[(working["Price_in_Lakhs"] > 0) & (working["Size_in_SqFt"] > 0)]
    user_rows = working.loc[working["_is_user_added"].astype(bool)]
    original_rows = working.loc[~working["_is_user_added"].astype(bool)]
    if len(original_rows) > 24_000:
        original_rows = original_rows.sample(24_000, random_state=41)
    working = pd.concat([original_rows, user_rows], ignore_index=True)
    if len(working) < 50:
        raise ValueError("Add or load at least 50 usable listings to fit the investor models.")

    X = working[MODEL_FEATURES].copy()
    y = working["Price_in_Lakhs"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    regressor = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("forest", ExtraTreesRegressor(
            n_estimators=140,
            max_features=0.9,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=41,
        )),
    ])
    regressor.fit(X_train, y_train)
    price_predictions = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, price_predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, price_predictions)))
    r2 = r2_score(y_test, price_predictions)

    low_cut, high_cut = np.quantile(y_train, [1 / 3, 2 / 3])
    if low_cut >= high_cut:
        low_cut, high_cut = np.quantile(y_train, [0.4, 0.8])

    def segment(values: pd.Series) -> pd.Series:
        labels = np.where(values <= low_cut, "Value", np.where(values <= high_cut, "Mid-market", "Premium"))
        return pd.Series(labels, index=values.index)

    y_train_segment = segment(y_train)
    y_test_segment = segment(y_test)
    stratify_values = y_train_segment if y_train_segment.value_counts().min() >= 2 else None
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_train, y_train_segment, test_size=0.2, random_state=41, stratify=stratify_values,
    )
    classifier = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("forest", ExtraTreesClassifier(
            n_estimators=140,
            max_features="sqrt",
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=41,
        )),
    ])
    classifier.fit(X_train_c, y_train_c)
    class_predictions = classifier.predict(X_test_c)
    class_accuracy = accuracy_score(y_test_c, class_predictions)
    class_f1 = f1_score(y_test_c, class_predictions, average="macro", zero_division=0)
    return {
        "regressor": regressor,
        "classifier": classifier,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "class_accuracy": class_accuracy,
        "class_f1": class_f1,
        "training_rows": len(working),
        "low_cut": low_cut,
        "high_cut": high_cut,
        "features": MODEL_FEATURES,
    }


extras_key = ""
if not extra_df.empty:
    extras_key = sha256(extra_df[REQUIRED_COLUMNS].to_csv(index=False).encode("utf-8")).hexdigest()
model_fingerprint = f"{source_signature}:{extras_key}"
model_error = None
models = None
try:
    models = train_models(analysis_all, model_fingerprint)
except (ValueError, KeyError) as exc:
    model_error = str(exc)


# -----------------------------------------------------------------------------
# Reusable plotting helpers and hero
# -----------------------------------------------------------------------------
def plot(fig, height: int = 390) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=50, b=12),
        font=dict(family="DM Sans, sans-serif", color="#29445E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def money_lakh(value: float) -> str:
    return f"₹{value:,.2f} L"


hero_b64 = ""
if HERO_PATH.exists():
    import base64
    hero_b64 = base64.b64encode(HERO_PATH.read_bytes()).decode("ascii")
hero_image = f'<img class="hero-img" src="data:image/svg+xml;base64,{hero_b64}" alt="Illustrated modern Indian residential skyline" />' if hero_b64 else ""
st.markdown(
    f"""
    <section class="hero-wrap">
      {hero_image}<div class="hero-shade"></div>
      <div class="hero-content">
        <div class="eyebrow">NEST · PROPERTY INVESTMENT STUDIO</div>
        <h1 class="hero-title">See the upside.<br/>Price the risk.</h1>
        <div class="hero-copy">Explore India's housing market, estimate a fair property price with machine learning, and test your own investment assumptions before you make a move.</div>
        <span class="hero-chip">📊 Market intelligence</span><span class="hero-chip">🤖 Regression + classification</span><span class="hero-chip">⚡ Live what-if analysis</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

avg_price = filtered["Price_in_Lakhs"].mean()
median_price = filtered["Price_in_Lakhs"].median()
median_psf = filtered["Calculated_Price_per_SqFt_Rs"].median()
added_count = int(filtered["_is_user_added"].sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings in view", f"{len(filtered):,}", help="Listings after applying the sidebar filters")
k2.metric("Average asking price", money_lakh(avg_price))
k3.metric("Median price / sq ft", f"₹{median_psf:,.0f}")
k4.metric("Investor-added listings", f"{added_count:,}", help="New listings added in this browser session")

market_tab, location_tab, strategy_tab, workspace_tab = st.tabs(
    ["🌐 Market pulse", "📍 Location intelligence", "🧠 Investment lab", "➕ Data workspace"]
)


# -----------------------------------------------------------------------------
# Market Pulse: distributions, price drivers, outliers, correlations
# -----------------------------------------------------------------------------
with market_tab:
    st.markdown('<div class="section-kicker">Market pulse</div>', unsafe_allow_html=True)
    st.header("A clearer read on the market")
    st.markdown("**Prices and property size**")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x="Price_in_Lakhs", nbins=42, marginal="box", color_discrete_sequence=[PALETTE[0]], labels={"Price_in_Lakhs": "Asking price (₹ lakhs)"}, title="Asking price distribution")
        plot(fig)
        st.caption(f"Median asking price: {money_lakh(median_price)}. The shape shows how concentrated or spread out the market is.")
    with right:
        fig = px.histogram(filtered, x="Size_in_SqFt", nbins=42, marginal="box", color_discrete_sequence=[PALETTE[1]], labels={"Size_in_SqFt": "Built-up area (sq ft)"}, title="Property size distribution")
        plot(fig)
        st.caption(f"Median home size: {filtered['Size_in_SqFt'].median():,.0f} sq ft.")

    st.markdown("**Price efficiency by home type**")
    by_type = filtered.groupby("Property_Type", as_index=False).agg(Median_Psf=("Calculated_Price_per_SqFt_Rs", "median"), Mean_Psf=("Calculated_Price_per_SqFt_Rs", "mean"), Listings=("Price_in_Lakhs", "size"))
    fig = px.box(filtered, x="Property_Type", y="Calculated_Price_per_SqFt_Rs", color="Property_Type", points=False, labels={"Property_Type": "Property type", "Calculated_Price_per_SqFt_Rs": "Calculated price / sq ft (₹)"}, title="Price per square foot by property type")
    plot(fig, 440)
    st.dataframe(by_type.sort_values("Median_Psf", ascending=False).style.format({"Median_Psf": "₹{:,.0f}", "Mean_Psf": "₹{:,.0f}", "Listings": "{:,.0f}"}), use_container_width=True, hide_index=True)

    st.markdown("**Does more space usually cost more?**")
    scatter = filtered.sample(min(6_000, len(filtered)), random_state=42)
    corr_size_price = filtered[["Size_in_SqFt", "Price_in_Lakhs"]].corr().iloc[0, 1]
    fig = px.scatter(scatter, x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type", opacity=.55, trendline="ols" if len(scatter) > 2 else None, hover_data=["City", "Locality", "BHK"], labels={"Size_in_SqFt": "Area (sq ft)", "Price_in_Lakhs": "Asking price (₹ lakhs)"}, title="Property size and asking price")
    plot(fig, 470)
    st.info(f"The size-price correlation in this selection is **{corr_size_price:.2f}**. A positive value means larger homes tend to have higher asking prices in this dataset; it is an association, not a guarantee.")

    st.markdown("**Outlier check**")
    outlier_stats = []
    for metric, col in [("Price / sq ft", "Calculated_Price_per_SqFt_Rs"), ("Property size", "Size_in_SqFt")]:
        q1, q3 = filtered[col].quantile([.25, .75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_stats.append({"Measure": metric, "Lower IQR bound": low, "Upper IQR bound": high, "Flagged listings": int(((filtered[col] < low) | (filtered[col] > high)).sum())})
    st.dataframe(pd.DataFrame(outlier_stats).style.format({"Lower IQR bound": "{:,.1f}", "Upper IQR bound": "{:,.1f}", "Flagged listings": "{:,.0f}"}), use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    with a:
        plot(px.box(filtered.sample(min(6_000, len(filtered)), random_state=9), y="Calculated_Price_per_SqFt_Rs", points="outliers", color_discrete_sequence=[PALETTE[2]], title="Price / sq ft outliers"), 340)
    with b:
        plot(px.box(filtered.sample(min(6_000, len(filtered)), random_state=9), y="Size_in_SqFt", points="outliers", color_discrete_sequence=[PALETTE[3]], title="Size outliers"), 340)
    st.caption("Outlier flags use the 1.5×IQR rule. Check unusual listings before treating them as errors.")

    st.markdown("**What moves together? Numeric feature correlation**")
    candidate_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Calculated_Price_per_SqFt_Rs", "Year_Built", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors"]
    corr_cols = [c for c in candidate_cols if c in filtered and filtered[c].nunique(dropna=True) > 1]
    if len(corr_cols) >= 2:
        corr_matrix = filtered[corr_cols].corr(min_periods=2).dropna(axis=0, how="all").dropna(axis=1, how="all")
        if len(corr_matrix.columns) >= 2:
            fig = px.imshow(corr_matrix, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto", title="Numeric feature correlation")
            plot(fig, max(510, 54 * len(corr_matrix.columns)))
            st.dataframe(corr_matrix.round(2), use_container_width=True)
            pairs = [(corr_matrix.iloc[i, j], corr_matrix.index[i], corr_matrix.columns[j]) for i in range(len(corr_matrix)) for j in range(i + 1, len(corr_matrix.columns)) if pd.notna(corr_matrix.iloc[i, j])]
            pairs.sort(key=lambda item: abs(item[0]), reverse=True)
            if pairs:
                st.caption("Strongest pairs: " + " · ".join(f"{a.replace('_', ' ')} ↔ {b.replace('_', ' ')} ({v:+.2f})" for v, a, b in pairs[:3]))
        else:
            st.info("There are not enough overlapping observations for a correlation matrix in this filter selection.")
    else:
        st.info("Choose a broader set of listings so at least two numeric features vary.")


# -----------------------------------------------------------------------------
# Location intelligence: state, city, locality, BHK, amenities and transit
# -----------------------------------------------------------------------------
with location_tab:
    st.markdown('<div class="section-kicker">Location intelligence</div>', unsafe_allow_html=True)
    st.header("Find a location that fits your strategy")
    state_psf = filtered.groupby("State", as_index=False).agg(Avg_Psf=("Calculated_Price_per_SqFt_Rs", "mean"), Listings=("Price_in_Lakhs", "size")).sort_values("Avg_Psf", ascending=False)
    city_price = filtered.groupby("City", as_index=False).agg(Avg_Price=("Price_in_Lakhs", "mean"), Listings=("Price_in_Lakhs", "size")).sort_values("Avg_Price", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(state_psf, x="State", y="Avg_Psf", color="Avg_Psf", color_continuous_scale=[[0, "#A7E9DF"], [.55, "#16B8A6"], [1, "#087B7A"]], hover_data=["Listings"], labels={"Avg_Psf": "Average price / sq ft (₹)"}, title="Average price per sq ft by state")
        plot(fig, 430)
    with c2:
        top_city = city_price.head(18).sort_values("Avg_Price")
        fig = px.bar(top_city, x="Avg_Price", y="City", orientation="h", color="Avg_Price", color_continuous_scale="Blues", hover_data=["Listings"], labels={"Avg_Price": "Average asking price (₹ lakhs)"}, title="Average asking price by city")
        plot(fig, 500)
    st.markdown("**Locality age and premium pockets**")
    locality_stats = filtered.groupby(["State", "City", "Locality"], as_index=False).agg(Median_Age=("Age_of_Property", "median"), Median_Psf=("Calculated_Price_per_SqFt_Rs", "median"), Listings=("Price_in_Lakhs", "size"))
    local_left, local_right = st.columns(2)
    with local_left:
        age_localities = locality_stats.dropna(subset=["Median_Age"]).nlargest(20, "Median_Age").copy()
        age_localities["Place"] = age_localities["Locality"] + " · " + age_localities["City"]
        fig = px.bar(age_localities.sort_values("Median_Age"), x="Median_Age", y="Place", orientation="h", color="Median_Age", color_continuous_scale="Sunset", hover_data=["State", "Listings"], labels={"Median_Age": "Median age (years)"}, title="Median property age by locality")
        plot(fig, 500)
    with local_right:
        expensive = locality_stats.loc[locality_stats["Listings"] >= 5].nlargest(15, "Median_Psf").sort_values("Median_Psf")
        expensive["Place"] = expensive["Locality"] + " · " + expensive["City"]
        fig = px.bar(expensive, x="Median_Psf", y="Place", orientation="h", color="Median_Psf", color_continuous_scale="Tealgrn", hover_data=["Listings"], labels={"Median_Psf": "Median price / sq ft (₹)"}, title="Higher-priced localities · 5+ listings")
        plot(fig, 500)
    with st.expander("View median age for every locality"):
        st.dataframe(locality_stats.sort_values("Median_Age", ascending=False).style.format({"Median_Age": "{:,.1f}", "Median_Psf": "₹{:,.0f}", "Listings": "{:,.0f}"}), use_container_width=True, hide_index=True)

    st.markdown("**Home mix and neighborhood signals**")
    bhk_city = filtered.groupby(["City", "BHK"], as_index=False).size().rename(columns={"size": "Listings"})
    top_cities = filtered["City"].value_counts().head(12).index
    bhk_city = bhk_city.loc[bhk_city["City"].isin(top_cities)]
    amenity_records = filtered[["Amenities", "Calculated_Price_per_SqFt_Rs"]].dropna(subset=["Amenities"]).assign(Amenities=lambda x: x["Amenities"].astype(str).str.split(",")).explode("Amenities")
    amenity_records["Amenities"] = amenity_records["Amenities"].str.strip()
    amenity_stats = amenity_records.loc[amenity_records["Amenities"].ne("")].groupby("Amenities", as_index=False).agg(Median_Psf=("Calculated_Price_per_SqFt_Rs", "median"), Listings=("Calculated_Price_per_SqFt_Rs", "size"))
    amenity_stats = amenity_stats.loc[amenity_stats["Listings"] >= 10].nlargest(12, "Median_Psf")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(bhk_city, x="City", y="Listings", color="BHK", barmode="stack", labels={"BHK": "Bedrooms"}, title="BHK mix across the busiest cities")
        plot(fig, 440)
    with col_b:
        fig = px.bar(amenity_stats.sort_values("Median_Psf"), x="Median_Psf", y="Amenities", orientation="h", color="Median_Psf", color_continuous_scale="Viridis", hover_data=["Listings"], labels={"Median_Psf": "Median price / sq ft (₹)"}, title="Price signal by amenity")
        plot(fig, 440)

    school_hospital_left, school_hospital_right = st.columns(2)
    for box, column, label in [(school_hospital_left, "Nearby_Schools", "Nearby schools"), (school_hospital_right, "Nearby_Hospitals", "Nearby hospitals")]:
        with box:
            facility = filtered.dropna(subset=[column]).groupby(column, as_index=False).agg(Median_Psf=("Calculated_Price_per_SqFt_Rs", "median"), Listings=("Calculated_Price_per_SqFt_Rs", "size"))
            fig = px.scatter(facility, x=column, y="Median_Psf", size="Listings", trendline="ols" if len(facility) > 2 else None, labels={column: label, "Median_Psf": "Median price / sq ft (₹)"}, title=f"{label} and price per sq ft")
            plot(fig, 360)
    transit = filtered.groupby("Public_Transport_Accessibility", as_index=False).agg(Median_Psf=("Calculated_Price_per_SqFt_Rs", "median"), Listings=("Price_in_Lakhs", "size"))
    fig = px.bar(transit, x="Public_Transport_Accessibility", y="Median_Psf", color="Public_Transport_Accessibility", text_auto=".0f", hover_data=["Listings"], labels={"Public_Transport_Accessibility": "Public transport access", "Median_Psf": "Median price / sq ft (₹)"}, title="Transport access and local price signal")
    plot(fig, 390)
    st.caption("Facility, amenity, and transit charts show associations in the listings. They do not isolate the effect of one feature.")

    st.markdown("**How features compare across prices**")
    feature_left, feature_right = st.columns(2)
    with feature_left:
        furn = filtered.groupby("Furnished_Status", as_index=False).agg(Median_Price=("Price_in_Lakhs", "median"), Listings=("Price_in_Lakhs", "size"))
        fig = px.bar(furn, x="Furnished_Status", y="Median_Price", color="Furnished_Status", hover_data=["Listings"], labels={"Furnished_Status": "Furnished status", "Median_Price": "Median price (₹ lakhs)"}, title="Price by furnished status")
        plot(fig, 360)
    with feature_right:
        facing = filtered.groupby("Facing", as_index=False).agg(Median_Psf=("Calculated_Price_per_SqFt_Rs", "median"), Listings=("Calculated_Price_per_SqFt_Rs", "size"))
        fig = px.bar(facing, x="Facing", y="Median_Psf", color="Facing", hover_data=["Listings"], labels={"Median_Psf": "Median price / sq ft (₹)"}, title="Price per sq ft by facing")
        plot(fig, 360)
    p1, p2 = st.columns(2)
    with p1:
        parking = filtered.groupby("Parking_Space", as_index=False).agg(Median_Price=("Price_in_Lakhs", "median"), Listings=("Price_in_Lakhs", "size"))
        plot(px.bar(parking, x="Parking_Space", y="Median_Price", color="Parking_Space", hover_data=["Listings"], title="Price by parking availability", labels={"Median_Price": "Median price (₹ lakhs)"}), 350)
    with p2:
        counts = filtered["Owner_Type"].value_counts().rename_axis("Owner type").reset_index(name="Listings")
        plot(px.bar(counts, x="Owner type", y="Listings", color="Owner type", text_auto=True, title="Listings by owner type"), 350)
    status = filtered["Availability_Status"].value_counts().rename_axis("Availability status").reset_index(name="Listings")
    plot(px.bar(status, x="Availability status", y="Listings", color="Availability status", text_auto=True, title="Listings by availability status"), 350)


# -----------------------------------------------------------------------------
# Investment Lab: investor-input scenario, estimated price, segment, yield, ROI
# -----------------------------------------------------------------------------
with strategy_tab:
    st.markdown('<div class="section-kicker">Investment lab</div>', unsafe_allow_html=True)
    st.header("Build a property scenario")
    st.write("Enter a property's features to estimate a market price and price segment. Add your own budget and rental assumptions to explore a simple return scenario.")
    if model_error:
        st.warning(f"Models could not be fitted: {model_error}")
    elif models:
        model_cols = st.columns(3)
        model_cols[0].metric("Regression · test MAE", money_lakh(models["mae"]), help="Average absolute price error on a held-out sample")
        model_cols[1].metric("Regression · test R²", f"{models['r2']:.2f}", help="Closer to 1 means more variation explained on the test split")
        model_cols[2].metric("Classification · macro F1", f"{models['class_f1']:.2f}", help="Balances performance across predicted price segments")
        st.caption(f"Models train on up to 24,000 original rows plus all investor-added listings in this session · {models['training_rows']:,} training rows after sampling. Classification identifies a feature-based price segment (Value, Mid-market, Premium).")

        markets = analysis_all
        states = sorted(markets["State"].dropna().astype(str).unique().tolist())
        properties = sorted(markets["Property_Type"].dropna().astype(str).unique().tolist())
        facing_options = sorted(markets["Facing"].dropna().astype(str).unique().tolist())
        furnished_options = sorted(markets["Furnished_Status"].dropna().astype(str).unique().tolist())
        transit_options = sorted(markets["Public_Transport_Accessibility"].dropna().astype(str).unique().tolist())
        parking_options = sorted(markets["Parking_Space"].dropna().astype(str).unique().tolist())
        security_options = sorted(markets["Security"].dropna().astype(str).unique().tolist())
        owner_options = sorted(markets["Owner_Type"].dropna().astype(str).unique().tolist())
        default_state = states[0] if states else "Unknown"
        default_property = properties[0] if properties else "Apartment"
        default_facing = facing_options[0] if facing_options else "Unknown"
        default_furnished = furnished_options[0] if furnished_options else "Unknown"
        default_transit = transit_options[0] if transit_options else "Unknown"
        default_parking = parking_options[0] if parking_options else "Unknown"
        default_security = security_options[0] if security_options else "Unknown"
        default_owner = owner_options[0] if owner_options else "Unknown"

        with st.form("investor_scenario"):
            st.subheader("Property profile")
            p1, p2, p3, p4 = st.columns(4)
            state = p1.selectbox("State", states, index=states.index(default_state))
            city = p2.text_input("City", value=str(markets["City"].mode().iloc[0]))
            locality = p3.text_input("Locality", value="New locality")
            property_type = p4.selectbox("Property type", properties, index=properties.index(default_property))
            p5, p6, p7, p8 = st.columns(4)
            bhk = p5.number_input("BHK", min_value=1, max_value=12, value=2, step=1)
            size = p6.number_input("Area (sq ft)", min_value=100, max_value=20_000, value=1_000, step=50)
            year_built = p7.number_input("Year built", min_value=1900, max_value=TODAY_YEAR + 2, value=TODAY_YEAR - 4, step=1)
            floor_no = p8.number_input("Floor number", min_value=0, max_value=150, value=3, step=1)
            p9, p10, p11, p12 = st.columns(4)
            total_floors = p9.number_input("Total floors", min_value=1, max_value=200, value=8, step=1)
            nearby_schools = p10.number_input("Nearby schools", min_value=0, max_value=100, value=4, step=1)
            nearby_hospitals = p11.number_input("Nearby hospitals", min_value=0, max_value=100, value=2, step=1)
            furnished = p12.selectbox("Furnished status", furnished_options, index=furnished_options.index(default_furnished))
            p13, p14, p15, p16 = st.columns(4)
            facing = p13.selectbox("Facing", facing_options, index=facing_options.index(default_facing))
            transit = p14.selectbox("Public transport", transit_options, index=transit_options.index(default_transit))
            parking = p15.selectbox("Parking", parking_options, index=parking_options.index(default_parking))
            security = p16.selectbox("Security", security_options, index=security_options.index(default_security))
            p17, p18 = st.columns(2)
            owner = p17.selectbox("Owner type", owner_options, index=owner_options.index(default_owner))
            amenities = p18.text_input("Amenities (comma separated)", value="Lift, Security, Gym")
            st.subheader("Your investment assumptions")
            q1, q2, q3, q4 = st.columns(4)
            budget = q1.number_input("Capital budget (₹ lakhs)", min_value=0.0, value=100.0, step=5.0)
            monthly_rent = q2.number_input("Expected monthly rent (₹)", min_value=0, value=25_000, step=2_500)
            closing_cost_pct = q3.number_input("Purchase costs (%)", min_value=0.0, max_value=30.0, value=7.0, step=.5)
            vacancy_pct = q4.slider("Expected vacancy (%)", min_value=0, max_value=50, value=5, step=1)
            q5, q6, q7 = st.columns(3)
            annual_maintenance = q5.number_input("Annual maintenance + tax (₹)", min_value=0, value=60_000, step=10_000)
            appreciation_pct = q6.number_input("Assumed annual appreciation (%)", min_value=-20.0, max_value=30.0, value=5.0, step=.5)
            holding_years = q7.slider("Holding period (years)", min_value=1, max_value=20, value=5, step=1)
            run_scenario = st.form_submit_button("✨ Estimate this investment", type="primary", use_container_width=True)

        if run_scenario:
            scenario = {
                "State": state, "City": city.strip() or "Unknown", "Locality": locality.strip() or "Unknown",
                "Property_Type": property_type, "BHK": bhk, "Size_in_SqFt": size,
                "Year_Built": year_built, "Age_of_Property": max(0, TODAY_YEAR - int(year_built)),
                "Floor_No": floor_no, "Total_Floors": total_floors, "Nearby_Schools": nearby_schools,
                "Nearby_Hospitals": nearby_hospitals, "Furnished_Status": furnished, "Facing": facing,
                "Public_Transport_Accessibility": transit, "Parking_Space": parking, "Security": security,
                "Owner_Type": owner, "Amenities": amenities,
            }
            scenario_frame = pd.DataFrame([scenario], columns=MODEL_FEATURES)
            estimate = float(models["regressor"].predict(scenario_frame)[0])
            classifier_price_segment = str(models["classifier"].predict(scenario_frame)[0])
            transformed = models["regressor"].named_steps["preprocessor"].transform(scenario_frame)
            trees = models["regressor"].named_steps["forest"].estimators_
            tree_estimates = np.array([tree.predict(transformed)[0] for tree in trees], dtype=float)
            low_estimate, high_estimate = np.quantile(tree_estimates, [.10, .90])
            assumptions = {
                "budget": budget, "monthly_rent": monthly_rent, "closing_cost_pct": closing_cost_pct,
                "vacancy_pct": vacancy_pct, "annual_maintenance": annual_maintenance,
                "appreciation_pct": appreciation_pct, "holding_years": holding_years,
            }
            st.session_state.scenario_result = {"scenario": scenario, "estimate": estimate, "segment": classifier_price_segment, "tree_range": (float(low_estimate), float(high_estimate)), "assumptions": assumptions}

        result = st.session_state.scenario_result
        if result:
            estimate = result["estimate"]
            assumptions = result["assumptions"]
            estimated_cost = estimate * (1 + assumptions["closing_cost_pct"] / 100)
            annual_gross_rent = assumptions["monthly_rent"] * 12 * (1 - assumptions["vacancy_pct"] / 100)
            annual_net_rent = annual_gross_rent - assumptions["annual_maintenance"]
            gross_yield = annual_gross_rent / (estimate * 100_000) * 100 if estimate > 0 else 0
            net_yield = annual_net_rent / (estimated_cost * 100_000) * 100 if estimated_cost > 0 else 0
            future_value = estimate * (1 + assumptions["appreciation_pct"] / 100) ** assumptions["holding_years"]
            cumulative_net_rent_lakh = annual_net_rent * assumptions["holding_years"] / 100_000
            projected_profit = future_value + cumulative_net_rent_lakh - estimated_cost
            projected_roi = projected_profit / estimated_cost * 100 if estimated_cost > 0 else 0
            st.markdown("---")
            st.subheader("Your live scenario readout")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Model-estimated property price", money_lakh(estimate))
            r2.metric("Estimated price / sq ft", f"₹{estimate * 100_000 / result['scenario']['Size_in_SqFt']:,.0f}")
            r3.metric("Predicted price segment", result["segment"])
            r4.metric("Budget after purchase costs", "Within budget" if estimated_cost <= assumptions["budget"] else "Over budget", delta=f"{estimated_cost - assumptions['budget']:+,.1f} L needed" if estimated_cost > assumptions["budget"] else f"{assumptions['budget'] - estimated_cost:,.1f} L buffer", delta_color="inverse" if estimated_cost > assumptions["budget"] else "normal")
            st.caption(f"ExtraTrees model estimate range (10th–90th percentile across trees): {money_lakh(result['tree_range'][0])}–{money_lakh(result['tree_range'][1])}. This is a model-spread indicator, not a formal confidence interval.")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Estimated all-in cost", money_lakh(estimated_cost))
            s2.metric("Gross rental yield", f"{gross_yield:.2f}%")
            s3.metric("Net yield after entered costs", f"{net_yield:.2f}%")
            s4.metric(f"Simple projected ROI · {assumptions['holding_years']}y", f"{projected_roi:.1f}%")
            value_path = pd.DataFrame({"Year": range(assumptions["holding_years"] + 1)})
            value_path["Estimated property value (₹ lakhs)"] = [estimate * (1 + assumptions["appreciation_pct"] / 100) ** year for year in value_path["Year"]]
            fig = px.area(value_path, x="Year", y="Estimated property value (₹ lakhs)", color_discrete_sequence=[PALETTE[0]], title="Illustrative property value path · based on your appreciation assumption")
            plot(fig, 340)
            st.warning("Scenario outputs are estimates, not investment advice. The return calculation uses your assumptions, simple compounded appreciation, entered rent, vacancy and operating costs. It excludes financing, tax, inflation, legal costs, and resale friction.")
        st.markdown("#### Model card")
        st.write("**Regression target:** asking price in lakhs. **Classification target:** a relative price band (Value, Mid-market, Premium) created from training-set price terciles. Models use a held-out split; displayed MAE, R², accuracy, and macro F1 are validation metrics, not guarantees for a specific property.")
    else:
        st.info("The investment models need at least 50 usable listings. Upload a larger dataset or add listings in the Data workspace.")


# -----------------------------------------------------------------------------
# Data workspace: new listings immediately join analytics and model training
# -----------------------------------------------------------------------------
with workspace_tab:
    st.markdown('<div class="section-kicker">Live data workspace</div>', unsafe_allow_html=True)
    st.header("Add a comparable. See the market update.")
    st.markdown('<div class="soft-note">When you submit a new listing, it is added to the charts and included in the next model fit for your current session. Download the updated CSV below to keep the listing outside this session.</div>', unsafe_allow_html=True)
    form_states = sorted(analysis_all["State"].dropna().astype(str).unique().tolist())
    form_cities = sorted(analysis_all["City"].dropna().astype(str).unique().tolist())
    form_types = sorted(analysis_all["Property_Type"].dropna().astype(str).unique().tolist())
    form_furnished = sorted(analysis_all["Furnished_Status"].dropna().astype(str).unique().tolist())
    form_facing = sorted(analysis_all["Facing"].dropna().astype(str).unique().tolist())
    form_transit = sorted(analysis_all["Public_Transport_Accessibility"].dropna().astype(str).unique().tolist())
    form_parking = sorted(analysis_all["Parking_Space"].dropna().astype(str).unique().tolist())
    form_security = sorted(analysis_all["Security"].dropna().astype(str).unique().tolist())
    form_owner = sorted(analysis_all["Owner_Type"].dropna().astype(str).unique().tolist())
    form_status = sorted(analysis_all["Availability_Status"].dropna().astype(str).unique().tolist())
    with st.form("add_comparable_listing", clear_on_submit=True):
        st.subheader("New comparable listing")
        f1, f2, f3, f4 = st.columns(4)
        new_state = f1.selectbox("State", form_states, key="new_state")
        new_city = f2.text_input("City", key="new_city", placeholder="e.g., Bengaluru")
        new_locality = f3.text_input("Locality", key="new_locality", placeholder="e.g., Whitefield")
        new_type = f4.selectbox("Property type", form_types, key="new_property_type")
        f5, f6, f7, f8 = st.columns(4)
        new_bhk = f5.number_input("BHK", 1, 12, 2, key="new_bhk")
        new_size = f6.number_input("Area (sq ft)", 100, 20_000, 1_000, 50, key="new_size")
        new_price = f7.number_input("Asking price (₹ lakhs)", 1.0, 100_000.0, 75.0, 5.0, key="new_price")
        new_year = f8.number_input("Year built", 1900, TODAY_YEAR + 2, TODAY_YEAR - 5, key="new_year")
        f9, f10, f11, f12 = st.columns(4)
        new_furn = f9.selectbox("Furnished status", form_furnished, key="new_furn")
        new_facing = f10.selectbox("Facing", form_facing, key="new_facing")
        new_transit = f11.selectbox("Public transport", form_transit, key="new_transit")
        new_parking = f12.selectbox("Parking", form_parking, key="new_parking")
        f13, f14, f15, f16 = st.columns(4)
        new_schools = f13.number_input("Nearby schools", 0, 100, 3, key="new_schools")
        new_hospitals = f14.number_input("Nearby hospitals", 0, 100, 2, key="new_hospitals")
        new_security = f15.selectbox("Security", form_security, key="new_security")
        new_owner = f16.selectbox("Owner type", form_owner, key="new_owner")
        f17, f18, f19, f20 = st.columns(4)
        new_floor = f17.number_input("Floor", 0, 150, 2, key="new_floor")
        new_total_floors = f18.number_input("Total floors", 1, 200, 8, key="new_total_floors")
        new_status = f19.selectbox("Availability", form_status, key="new_status")
        new_amenities = f20.text_input("Amenities", key="new_amenities", placeholder="Lift, Gym, Garden")
        add_listing = st.form_submit_button("＋ Add listing to this session", type="primary", use_container_width=True)

    if add_listing:
        if not new_city.strip() or not new_locality.strip():
            st.error("Enter both a city and a locality to add the listing.")
        elif new_size <= 0 or new_price <= 0:
            st.error("Area and price must be greater than zero.")
        else:
            next_id = len(analysis_all) + 1
            new_row = {
                "ID": next_id, "State": new_state, "City": new_city.strip(), "Locality": new_locality.strip(),
                "Property_Type": new_type, "BHK": new_bhk, "Size_in_SqFt": new_size, "Price_in_Lakhs": new_price,
                "Price_per_SqFt": new_price * 100_000 / new_size, "Year_Built": new_year,
                "Furnished_Status": new_furn, "Floor_No": new_floor, "Total_Floors": new_total_floors,
                "Age_of_Property": max(0, TODAY_YEAR - int(new_year)), "Nearby_Schools": new_schools,
                "Nearby_Hospitals": new_hospitals, "Public_Transport_Accessibility": new_transit,
                "Parking_Space": new_parking, "Security": new_security, "Amenities": new_amenities,
                "Facing": new_facing, "Owner_Type": new_owner, "Availability_Status": new_status,
                "_is_user_added": True,
            }
            st.session_state.new_listings = pd.concat([st.session_state.new_listings, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Listing added. The dashboard and the next model fit now include it.")
            st.rerun()

    if not extra_df.empty:
        st.markdown(f"#### Investor-added records · {len(extra_df):,}")
        st.dataframe(extra_df.drop(columns=["_is_user_added"], errors="ignore").tail(100), use_container_width=True, hide_index=True)
        csv_download = analysis_all.drop(columns=["_is_user_added"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download full updated CSV", data=csv_download, file_name="india_housing_prices_updated.csv", mime="text/csv", type="primary")
        if st.button("Clear investor-added records from this session"):
            st.session_state.new_listings = pd.DataFrame(columns=REQUIRED_COLUMNS + ["ID", "_is_user_added"])
            st.session_state.scenario_result = None
            st.rerun()
    else:
        st.info("No session listings added yet. Add one above or upload an updated CSV in the sidebar.")

    st.markdown("#### Data quality snapshot")
    quality_a, quality_b, quality_c = st.columns(3)
    quality_a.metric("Rows available", f"{len(analysis_all):,}")
    quality_b.metric("Usable rows", f"{len(analysis_all.dropna(subset=['Price_in_Lakhs', 'Size_in_SqFt'])):,}")
    quality_c.metric("Dataset fields", f"{len(source_df.columns):,}")
    st.caption("Streamlit Cloud sessions are temporary. For a durable dataset update, download the updated CSV and replace the repository's CSV, then commit and push it to GitHub.")

st.markdown("---")
st.caption("NEST Investor Studio · Price per sq ft is calculated as asking price in lakhs × 100,000 ÷ area in sq ft. Model and return outputs are exploratory estimates based on the dataset and user assumptions.")
