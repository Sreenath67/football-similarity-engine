import os
import ast
import glob
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="⚽ Football Similarity Engine", layout="wide")

EXPECTED_FILES = [
    "AtMid_Wingers.csv",
    "CenterBacks.csv",
    "Forwards.csv",
    "FullBacks.csv",
    "GoalKeepers.csv",
    "Midfielders.csv",
]
DATA_FOLDER = "data"  

# ----------------------------
# Helpers
# ----------------------------
def parse_list_cell(x):
    """
    Safely parse a Python-list-like string into a list.
    Handles NaNs and already-parsed lists.
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []

def clean_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()

def ensure_2d_array(vectors: List[List[float]]) -> np.ndarray:
    # Replace None/empty vectors with zeros of the most common length
    lengths = [len(v) for v in vectors if isinstance(v, list)]
    if not lengths:
        return np.array([])
    target_len = max(set(lengths), key=lengths.count)
    fixed = []
    for v in vectors:
        if not isinstance(v, list):
            v = []
        if len(v) == target_len:
            fixed.append(v)
        else:
            # pad/truncate to target_len
            v = (v + [0.0] * target_len)[:target_len]
            fixed.append(v)
    return np.array(fixed, dtype=float)

def build_position_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data(show_spinner=True)
def load_combined_df(data_folder: str) -> pd.DataFrame:
    paths = []
    for fname in EXPECTED_FILES:
        p = os.path.join(data_folder, fname)
        if os.path.exists(p):
            paths.append(p)

    # also allow additional CSVs placed in the folder (flexible)
    for p in glob.glob(os.path.join(data_folder, "*.csv")):
        if p not in paths:
            paths.append(p)

    if not paths:
        return pd.DataFrame(columns=["Name", "Position", "Attribute Vector", "Percentiles"])

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, encoding="utf-8", engine="python")
        except Exception:
            df = pd.read_csv(p, encoding_errors="ignore", engine="python")

        # Try to find list-like columns by common names
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]

        # Map potential variants
        attr_col = None
        pct_col = None
        for cand in ["Attribute Vector", "AttributeVector", "Attributes", "Attribute_List"]:
            if cand in df.columns:
                attr_col = cand
                break
        for cand in ["Percentiles", "Percentile Vector", "Percents", "Percentile_List"]:
            if cand in df.columns:
                pct_col = cand
                break

        # Fallback: if not found, create empty
        if attr_col is None:
            df["Attribute Vector"] = [[] for _ in range(len(df))]
        else:
            df["Attribute Vector"] = df[attr_col].apply(parse_list_cell)

        if pct_col is None:
            df["Percentiles"] = [[] for _ in range(len(df))]
        else:
            df["Percentiles"] = df[pct_col].apply(parse_list_cell)

        if "Name" not in df.columns:
            # try a common alternative
            name_col = [c for c in df.columns if c.lower() == "player" or "name" in c.lower()]
            if name_col:
                df["Name"] = df[name_col[0]]
            else:
                df["Name"] = ""

        df["Name"] = df["Name"].apply(clean_name)
        df["Position"] = build_position_from_filename(p)
        frames.append(df[["Name", "Position", "Attribute Vector", "Percentiles"]])

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Name", "Position"])
    # Create a display label to disambiguate duplicates across positions
    combined["Display"] = combined["Name"] + " — " + combined["Position"]
    return combined

def compute_similarity(
    target_vec: List[float],
    candidate_vecs: List[List[float]]
) -> np.ndarray:
    target = np.array(target_vec, dtype=float).reshape(1, -1)
    cand = ensure_2d_array(candidate_vecs)
    if cand.size == 0 or target.size == 0:
        return np.zeros(len(candidate_vecs))
    # pad target to same length as candidates if needed
    if target.shape[1] != cand.shape[1]:
        max_len = max(target.shape[1], cand.shape[1])
        def pad(v, L): return np.pad(v, ((0,0),(0, L - v.shape[1])), mode="constant") if v.shape[1] < L else v[:, :L]
        target = pad(target, max_len)
        cand = pad(cand, max_len)
    sims = cosine_similarity(target, cand).flatten()
    return sims

def get_similar(
    df: pd.DataFrame,
    selected_idx: int,
    top_n: int,
    use_percentiles: bool,
    restrict_same_position: bool
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Name", "Position", "Similarity"])

    row = df.iloc[selected_idx]
    source_col = "Percentiles" if use_percentiles else "Attribute Vector"
    target_vec = row[source_col]

    if restrict_same_position:
        subset = df[df["Position"] == row["Position"]].reset_index(drop=True)
    else:
        subset = df.reset_index(drop=True)

    vectors = subset[source_col].tolist()
    sims = compute_similarity(target_vec, vectors)

    # Exclude exact row if it is within subset
    # We identify by Name+Position to be safe
    mask_self = (subset["Name"] == row["Name"]) & (subset["Position"] == row["Position"])
    sims = np.where(mask_self.values, -1.0, sims)

    top_idx = np.argsort(sims)[::-1][:top_n]
    out = subset.iloc[top_idx][["Name", "Position"]].copy()
    out.insert(2, "Similarity", sims[top_idx])
    return out.reset_index(drop=True)

# ----------------------------
# UI
# ----------------------------
st.title("⚽ Football Similarity Engine")

with st.expander("ℹ️ About this app", expanded=False):
    st.markdown(
        """
This app finds **similar football players** based on numerical feature vectors.
It supports:
- **Overall similarity** (across all positions)
- **Position-restricted similarity** (within the current player's position)

You can choose to compare using **raw attribute vectors** or **percentile vectors** (if available).
        """
    )

# Load data
combined_df = load_combined_df(DATA_FOLDER)

if combined_df.empty:
    st.warning(
        f"No CSVs found. Please add your files to `{DATA_FOLDER}/` "
        f"(e.g., {', '.join(EXPECTED_FILES)}) and refresh."
    )
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.subheader("Select Player & Options")
    player_options = combined_df["Display"].tolist()
    selected_display = st.selectbox("Player", player_options, index=0)
    selected_idx = combined_df.index[combined_df["Display"] == selected_display][0]

    top_n = st.slider("Number of similar players", min_value=1, max_value=20, value=5, step=1)

    compare_basis = st.radio(
        "Compare using",
        ["Attribute Vector", "Percentiles"],
        index=0,
        help="If Percentiles is selected but missing for some players, zeros will be used."
    )
    use_percentiles = (compare_basis == "Percentiles")

with right:
    st.subheader("Results")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("🔁 Similar (Same Position)")
        same_pos_df = get_similar(
            combined_df,
            selected_idx,
            top_n=top_n,
            use_percentiles=use_percentiles,
            restrict_same_position=True
        )
        st.dataframe(same_pos_df, use_container_width=True)

    with col2:
        st.caption("🌐 Similar (Overall)")
        overall_df = get_similar(
            combined_df,
            selected_idx,
            top_n=top_n,
            use_percentiles=use_percentiles,
            restrict_same_position=False
        )
        st.dataframe(overall_df, use_container_width=True)

st.markdown("---")
st.markdown(
    "If you see duplicate names in different positions, "
    "use the **Name — Position** selector to disambiguate."
)
