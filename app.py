import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="⚽ Football Similarity Engine", layout="wide")
DATA_FOLDER = "data"

# ----------------------------
# Helpers
# ----------------------------
def parse_list_cell(x):
    """Convert string representation of list to Python list"""
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except:
        return []

def clean_name(s):
    return str(s).strip()

def ensure_2d_array(vectors):
    """Make sure all vectors are same length (pad with zeros)"""
    lengths = [len(v) for v in vectors if isinstance(v, list)]
    if not lengths:
        return np.array([])
    target_len = max(set(lengths), key=lengths.count)
    fixed = [(v + [0.0] * target_len)[:target_len] if isinstance(v, list) else [0.0] * target_len for v in vectors]
    return np.array(fixed, dtype=float)

def compute_similarity(target_vec, candidate_vecs):
    """Cosine similarity between target and candidates"""
    target = np.array(target_vec, dtype=float).reshape(1, -1)
    cand = ensure_2d_array(candidate_vecs)
    if cand.size == 0 or target.size == 0:
        return np.zeros(len(candidate_vecs))
    if target.shape[1] != cand.shape[1]:
        max_len = max(target.shape[1], cand.shape[1])
        target = np.pad(target, ((0,0),(0, max_len - target.shape[1])), mode="constant")
        cand = np.pad(cand, ((0,0),(0, max_len - cand.shape[1])), mode="constant")
    return cosine_similarity(target, cand).flatten()

# ----------------------------
# Load Data (separate GK and others)
# ----------------------------
@st.cache_data
def load_data():
    # Load outfield players (all except GoalKeepers.csv)
    dfs = []
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".csv") and f != "GoalKeepers.csv":
            df = pd.read_csv(os.path.join(DATA_FOLDER, f))
            df.columns = [c.strip() for c in df.columns]
            if "Attribute Vector" in df.columns:
                df["Attribute Vector"] = df["Attribute Vector"].apply(parse_list_cell)
            df["Name"] = df.get("Name", df.iloc[:,0]).apply(clean_name)
            df["Position"] = os.path.splitext(f)[0]
            dfs.append(df[["Name","Position","Attribute Vector"]])
    outfield = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Load goalkeepers separately
    gk_path = os.path.join(DATA_FOLDER, "GoalKeepers.csv")
    gk_df = pd.DataFrame()
    if os.path.exists(gk_path):
        gk_df = pd.read_csv(gk_path)
        gk_df.columns = [c.strip() for c in gk_df.columns]
        if "Attribute Vector" in gk_df.columns:
            gk_df["Attribute Vector"] = gk_df["Attribute Vector"].apply(parse_list_cell)
        gk_df["Name"] = gk_df.get("Name", gk_df.iloc[:,0]).apply(clean_name)
        gk_df["Position"] = "GoalKeeper"

    return outfield, gk_df

# ----------------------------
# Similarity Finder
# ----------------------------
def get_similar(df, selected_idx, top_n):
    row = df.iloc[selected_idx]
    target_vec = row["Attribute Vector"]
    vectors = df["Attribute Vector"].tolist()
    sims = compute_similarity(target_vec, vectors)
    sims[selected_idx] = -1.0  # exclude self
    top_idx = np.argsort(sims)[::-1][:top_n]
    out = df.iloc[top_idx][["Name","Position"]].copy()
    out.insert(2, "Similarity", sims[top_idx])
    return out.reset_index(drop=True)

# ----------------------------
# UI
# ----------------------------
st.title("⚽ Football Similarity Engine")

outfield_df, gk_df = load_data()

if outfield_df.empty and gk_df.empty:
    st.warning("No player CSVs found in 'data/' folder.")
    st.stop()

player_type = st.radio("Choose Player Type", ["Outfield", "GoalKeeper"])

df = outfield_df if player_type == "Outfield" else gk_df
if df.empty:
    st.warning(f"No data found for {player_type}s.")
    st.stop()

player_options = df["Name"].tolist()
selected_name = st.selectbox("Select Player", player_options, index=0)
selected_idx = df.index[df["Name"] == selected_name][0]
top_n = st.slider("Number of similar players", 1, 20, 5)

results = get_similar(df, selected_idx, top_n)
st.subheader("Similar Players")
st.dataframe(results, use_container_width=True)
