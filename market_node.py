# demo_market_node_final.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ============================================================
# 0. Config
# ============================================================

MARKET_FILES = [
    ("data/rawdata/2024_q3.csv", 2024, 3),
    ("data/rawdata/2024_q4.csv", 2024, 4),
    ("data/rawdata/2025_q1.csv", 2025, 1),
    ("data/rawdata/2025_q2.csv", 2025, 2),
]

AIRPORTS_PATH = "data/rawdata/airports.dat.txt"

# Split options:
# "random_70_10_20" = standard model comparison split
# "temporal"        = train 2024 Q3/Q4, val 2025 Q1, test 2025 Q2
SPLIT_MODE = "random_70_10_20"

# Optional raw row sampling before aggregation.
# Keep None for main run. Use 0.20 for faster debugging.
ROW_SAMPLE_FRAC = None

# Market-node caps after aggregation.
# For debugging: 70_000 / 10_000 / 20_000
# For stronger run: 140_000 / 20_000 / 40_000
# For heavier run: 350_000 / 50_000 / 100_000
MAX_TRAIN_NODES = 350_000
MAX_VAL_NODES = 50_000
MAX_TEST_NODES = 100_000

RANDOM_STATE = 42

# Graph
K_NEIGHBORS = 2

# Training
MAX_EPOCHS_MLP = 250
MAX_EPOCHS_GNN = 250
PATIENCE = 30

MLP_LR = 5e-4
GNN_LR = 8e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 128
MLP_DROPOUT = 0.15
GNN_DROPOUT = 0.15

RUN_RANDOM_EDGE_CONTROL = True


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_dollars(y_true_log, y_pred_log, label=""):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true_log, y_pred_log)

    print(
        f"{label:<12} "
        f"RMSE: ${rmse:8.2f} | "
        f"MAE: ${mae:8.2f} | "
        f"MedAE: ${medae:8.2f} | "
        f"R2(log): {r2:7.3f}"
    )

    return rmse, mae, medae, r2


def print_split_by_period(df, masks):
    rows = []

    for split_name, mask in masks.items():
        temp = (
            df.loc[mask]
            .groupby(["YEAR", "QUARTER"])
            .size()
            .reset_index(name="nodes")
        )
        temp["split"] = split_name
        rows.append(temp)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["YEAR", "QUARTER", "split"])
    print(out)


def stratified_cap_mask(df, base_mask, max_rows, strata_cols, seed):
    idx_all = np.where(base_mask)[0]

    if len(idx_all) <= max_rows:
        out = np.zeros(len(df), dtype=bool)
        out[idx_all] = True
        return out

    rng = np.random.default_rng(seed)
    frac = max_rows / len(idx_all)

    selected = []

    temp = df.loc[base_mask, strata_cols].copy()
    temp["_row_idx"] = idx_all

    for _, g in temp.groupby(strata_cols, observed=True):
        ids = g["_row_idx"].to_numpy()
        n_take = int(np.floor(len(ids) * frac))

        if len(ids) > 0 and n_take == 0:
            n_take = 1

        n_take = min(n_take, len(ids))

        if n_take > 0:
            selected.extend(rng.choice(ids, size=n_take, replace=False).tolist())

    selected = np.array(selected, dtype=int)

    if len(selected) > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False)

    elif len(selected) < max_rows:
        selected_set = set(selected.tolist())
        remaining = np.array([i for i in idx_all if i not in selected_set], dtype=int)

        need = max_rows - len(selected)
        if len(remaining) > 0:
            add = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
            selected = np.concatenate([selected, add])

    out = np.zeros(len(df), dtype=bool)
    out[selected] = True
    return out


def build_random_edges(num_nodes, num_edges, seed=42):
    rng = np.random.default_rng(seed)
    edge_set = set()

    batch_size = max(num_edges * 2, 10000)

    while len(edge_set) < num_edges:
        src = rng.integers(0, num_nodes, size=batch_size)
        dst = rng.integers(0, num_nodes, size=batch_size)

        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        for s, d in zip(src, dst):
            edge_set.add((int(s), int(d)))
            if len(edge_set) >= num_edges:
                break

    edges = list(edge_set)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


set_seed(RANDOM_STATE)


# ============================================================
# 2. Load four DB1B market files
# ============================================================

print("Loading four DB1B market files...")

required_cols = [
    "ORIGIN",
    "DEST",
    "REPORTING_CARRIER",
    "TICKET_CARRIER",
    "OPERATING_CARRIER",
    "BULK_FARE",
    "PASSENGERS",
    "MARKET_FARE",
    "MARKET_DISTANCE",
    "NONSTOP_MILES",
    "MKT_GEO_TYPE",
]

optional_cols = [
    "YEAR",
    "QUARTER",
    "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID",
    "ORIGIN_LAT",
    "ORIGIN_LON",
    "DEST_LAT",
    "DEST_LON",
]

frames = []

for path, year, quarter in MARKET_FILES:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in required_cols if c not in header]

    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    usecols = [c for c in required_cols + optional_cols if c in header]

    temp = pd.read_csv(path, usecols=usecols, low_memory=False)

    temp["YEAR"] = year
    temp["QUARTER"] = quarter

    print(f"Loaded {path}: {temp.shape} as {year} Q{quarter}")
    frames.append(temp)

df = pd.concat(frames, ignore_index=True)

print(f"\nCombined raw shape: {df.shape}")
print("Periods:")
print(df[["YEAR", "QUARTER"]].drop_duplicates().sort_values(["YEAR", "QUARTER"]))


# ============================================================
# 3. Basic cleaning
# ============================================================

df = df[
    (df["BULK_FARE"] == 0)
    & (df["MKT_GEO_TYPE"] == 2)
    & (df["MARKET_FARE"] >= 20)
    & (df["MARKET_DISTANCE"] > 0)
    & (df["NONSTOP_MILES"] > 0)
].copy()

df = df.dropna(subset=[
    "YEAR",
    "QUARTER",
    "ORIGIN",
    "DEST",
    "TICKET_CARRIER",
    "PASSENGERS",
    "MARKET_FARE",
    "MARKET_DISTANCE",
    "NONSTOP_MILES",
])

df["YEAR"] = df["YEAR"].astype(int)
df["QUARTER"] = df["QUARTER"].astype(int)

df["ORIGIN"] = df["ORIGIN"].astype(str)
df["DEST"] = df["DEST"].astype(str)
df["TICKET_CARRIER"] = df["TICKET_CARRIER"].astype(str)

df["PASSENGERS"] = df["PASSENGERS"].clip(lower=1e-6)
df["fare_x_pax"] = df["MARKET_FARE"] * df["PASSENGERS"]

print(f"After filters: {df.shape}")


# ============================================================
# 4. Join airport coordinates if needed
# ============================================================

has_geo = all(c in df.columns for c in ["ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON"])

if has_geo:
    print("\nUsing existing coordinate columns.")
    df = df.rename(columns={
        "ORIGIN_LAT": "origin_lat",
        "ORIGIN_LON": "origin_lon",
        "DEST_LAT": "dest_lat",
        "DEST_LON": "dest_lon",
    })

else:
    print("\nJoining coordinates from airports.csv...")

    if not os.path.exists(AIRPORTS_PATH):
        raise FileNotFoundError(f"Missing airport file: {AIRPORTS_PATH}")

    airports = pd.read_csv(
        AIRPORTS_PATH, header=None, low_memory=False,
        names=["id","name","city","country","iata_code","icao",
               "latitude_deg","longitude_deg","altitude","timezone",
               "dst","tz_db","type","source"],
    )

    coords = (
        airports[
            airports["iata_code"].notna()
            & airports["iata_code"].ne("\\N")
            & airports["latitude_deg"].notna()
            & airports["longitude_deg"].notna()
        ][["iata_code", "latitude_deg", "longitude_deg"]]
        .drop_duplicates("iata_code")
        .copy()
    )

    coords["iata_code"] = coords["iata_code"].astype(str)

    df = df.merge(
        coords.rename(columns={
            "iata_code": "ORIGIN",
            "latitude_deg": "origin_lat",
            "longitude_deg": "origin_lon",
        }),
        on="ORIGIN",
        how="left",
    )

    df = df.merge(
        coords.rename(columns={
            "iata_code": "DEST",
            "latitude_deg": "dest_lat",
            "longitude_deg": "dest_lon",
        }),
        on="DEST",
        how="left",
    )

print("Missing coordinate counts:")
print(df[["origin_lat", "origin_lon", "dest_lat", "dest_lon"]].isna().sum())

before_geo_drop = len(df)
df = df.dropna(subset=["origin_lat", "origin_lon", "dest_lat", "dest_lon"]).copy()
print(f"Dropped rows missing coordinates: {before_geo_drop - len(df):,}")
print(f"After coordinate handling: {df.shape}")


# ============================================================
# 5. Optional row-level sample before aggregation
# ============================================================

if ROW_SAMPLE_FRAC is not None:
    print(f"\nApplying initial row sample before aggregation: {ROW_SAMPLE_FRAC:.0%}")

    before = len(df)

    df = (
        df.groupby(["YEAR", "QUARTER", "ORIGIN"], group_keys=False)
        .sample(frac=ROW_SAMPLE_FRAC, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print(f"Before initial sampling: {before:,}")
    print(f"After initial sampling:  {len(df):,}")


# ============================================================
# 6. Aggregate to OD-quarter market nodes
# ============================================================

print("\nAggregating to OD-quarter market nodes...")

market_keys = ["ORIGIN", "DEST", "YEAR", "QUARTER"]

market_agg = (
    df.groupby(market_keys, observed=True)
    .agg(
        passengers_sum=("PASSENGERS", "sum"),
        n_records=("MARKET_FARE", "size"),
        fare_x_pax_sum=("fare_x_pax", "sum"),
        distance_mean=("MARKET_DISTANCE", "mean"),
        nonstop_miles_mean=("NONSTOP_MILES", "mean"),
        origin_lat=("origin_lat", "first"),
        origin_lon=("origin_lon", "first"),
        dest_lat=("dest_lat", "first"),
        dest_lon=("dest_lon", "first"),
    )
    .reset_index()
)

market_agg["fare_weighted_mean"] = (
    market_agg["fare_x_pax_sum"] / market_agg["passengers_sum"]
)

market_agg["log_fare"] = np.log(market_agg["fare_weighted_mean"])

print(f"Initial market nodes: {len(market_agg):,}")


# ============================================================
# 7. Competition features
# ============================================================

carrier_agg = (
    df.groupby(market_keys + ["TICKET_CARRIER"], observed=True)
    .agg(carrier_passengers=("PASSENGERS", "sum"))
    .reset_index()
)

total_pax = (
    carrier_agg.groupby(market_keys, observed=True)["carrier_passengers"]
    .sum()
    .reset_index(name="market_passengers")
)

carrier_agg = carrier_agg.merge(total_pax, on=market_keys, how="left")
carrier_agg["carrier_share"] = (
    carrier_agg["carrier_passengers"] / carrier_agg["market_passengers"]
)

competition = (
    carrier_agg.groupby(market_keys, observed=True)
    .agg(
        num_carriers=("TICKET_CARRIER", "nunique"),
        hhi=("carrier_share", lambda x: np.sum(np.square(x))),
        max_carrier_share=("carrier_share", "max"),
    )
    .reset_index()
)

market = market_agg.merge(competition, on=market_keys, how="left")

market["num_carriers"] = market["num_carriers"].fillna(1)
market["hhi"] = market["hhi"].fillna(1.0)
market["max_carrier_share"] = market["max_carrier_share"].fillna(1.0)

del carrier_agg
del total_pax
del competition


# ============================================================
# 8. Structural features
# ============================================================

origin_stats = (
    market.groupby(["YEAR", "QUARTER", "ORIGIN"], observed=True)
    .agg(
        origin_market_count=("DEST", "nunique"),
        origin_total_passengers=("passengers_sum", "sum"),
        origin_avg_distance=("distance_mean", "mean"),
    )
    .reset_index()
)

dest_stats = (
    market.groupby(["YEAR", "QUARTER", "DEST"], observed=True)
    .agg(
        dest_market_count=("ORIGIN", "nunique"),
        dest_total_passengers=("passengers_sum", "sum"),
        dest_avg_distance=("distance_mean", "mean"),
    )
    .reset_index()
)

market = market.merge(origin_stats, on=["YEAR", "QUARTER", "ORIGIN"], how="left")
market = market.merge(dest_stats, on=["YEAR", "QUARTER", "DEST"], how="left")


# ============================================================
# 9. Lag features
# ============================================================

print("\nAdding route lag features...")

market = market.sort_values(["ORIGIN", "DEST", "YEAR", "QUARTER"]).reset_index(drop=True)

market["prev_log_fare"] = market.groupby(["ORIGIN", "DEST"])["log_fare"].shift(1)
market["prev_log_passengers"] = market.groupby(["ORIGIN", "DEST"])["passengers_sum"].shift(1)
market["prev_hhi"] = market.groupby(["ORIGIN", "DEST"])["hhi"].shift(1)
market["has_prev"] = market["prev_log_fare"].notna().astype(int)

market["prev_log_passengers"] = np.log1p(market["prev_log_passengers"])


# ============================================================
# 10. Feature engineering
# ============================================================

market["log_distance"] = np.log1p(market["distance_mean"])
market["log_passengers"] = np.log1p(market["passengers_sum"])
market["log_n_records"] = np.log1p(market["n_records"])

market["sin_q"] = np.sin(2 * np.pi * (market["QUARTER"] - 1) / 4)
market["cos_q"] = np.cos(2 * np.pi * (market["QUARTER"] - 1) / 4)

market["log_origin_total_passengers"] = np.log1p(market["origin_total_passengers"])
market["log_dest_total_passengers"] = np.log1p(market["dest_total_passengers"])

market["log_origin_market_count"] = np.log1p(market["origin_market_count"])
market["log_dest_market_count"] = np.log1p(market["dest_market_count"])

market["log_origin_avg_distance"] = np.log1p(market["origin_avg_distance"])
market["log_dest_avg_distance"] = np.log1p(market["dest_avg_distance"])

market["route_lat_diff"] = market["dest_lat"] - market["origin_lat"]
market["route_lon_diff"] = market["dest_lon"] - market["origin_lon"]


# ============================================================
# 11. Split
# ============================================================

print(f"\nSPLIT_MODE = {SPLIT_MODE}")

if SPLIT_MODE == "temporal":
    train_mask_np = market["YEAR"].eq(2024).values
    val_mask_np = ((market["YEAR"] == 2025) & (market["QUARTER"] == 1)).values
    test_mask_np = ((market["YEAR"] == 2025) & (market["QUARTER"] == 2)).values

    print("Temporal split:")
    print("Train = 2024 Q3 + Q4")
    print("Val   = 2025 Q1")
    print("Test  = 2025 Q2")

elif SPLIT_MODE == "random_70_10_20":
    rng = np.random.default_rng(RANDOM_STATE)

    train_mask_np = np.zeros(len(market), dtype=bool)
    val_mask_np = np.zeros(len(market), dtype=bool)
    test_mask_np = np.zeros(len(market), dtype=bool)

    print("Random stratified market split: 70% train / 10% val / 20% test")

    for _, group_idx in market.groupby(["YEAR", "QUARTER", "ORIGIN"], observed=True).groups.items():
        idx = np.array(list(group_idx))
        rng.shuffle(idx)

        n = len(idx)
        train_end = int(0.70 * n)
        val_end = int(0.80 * n)

        train_mask_np[idx[:train_end]] = True
        val_mask_np[idx[train_end:val_end]] = True
        test_mask_np[idx[val_end:]] = True

else:
    raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

print("\nBefore node caps:")
print(f"Train nodes: {train_mask_np.sum():,}")
print(f"Val nodes:   {val_mask_np.sum():,}")
print(f"Test nodes:  {test_mask_np.sum():,}")

print("\nSplit by period before caps:")
print_split_by_period(market, {
    "train": train_mask_np,
    "val": val_mask_np,
    "test": test_mask_np,
})


# ============================================================
# 12. Apply node caps
# ============================================================

print("\nApplying market-node caps...")

train_keep = stratified_cap_mask(
    market,
    train_mask_np,
    max_rows=MAX_TRAIN_NODES,
    strata_cols=["YEAR", "QUARTER", "ORIGIN"],
    seed=RANDOM_STATE + 10,
)

val_keep = stratified_cap_mask(
    market,
    val_mask_np,
    max_rows=MAX_VAL_NODES,
    strata_cols=["YEAR", "QUARTER", "ORIGIN"],
    seed=RANDOM_STATE + 20,
)

test_keep = stratified_cap_mask(
    market,
    test_mask_np,
    max_rows=MAX_TEST_NODES,
    strata_cols=["YEAR", "QUARTER", "ORIGIN"],
    seed=RANDOM_STATE + 30,
)

keep_mask = train_keep | val_keep | test_keep

market = market.loc[keep_mask].copy().reset_index(drop=True)

train_mask_np = train_keep[keep_mask]
val_mask_np = val_keep[keep_mask]
test_mask_np = test_keep[keep_mask]

market["node_id"] = np.arange(len(market))

print("\nAfter node caps:")
print(f"Train nodes: {train_mask_np.sum():,}")
print(f"Val nodes:   {val_mask_np.sum():,}")
print(f"Test nodes:  {test_mask_np.sum():,}")
print(f"Total nodes: {len(market):,}")

print("\nSplit by period after caps:")
print_split_by_period(market, {
    "train": train_mask_np,
    "val": val_mask_np,
    "test": test_mask_np,
})

if train_mask_np.sum() == 0 or val_mask_np.sum() == 0 or test_mask_np.sum() == 0:
    raise ValueError("Split failed: one of train/val/test is empty.")


# ============================================================
# 13. Fill lag features using train values
# ============================================================

train_mean_log_fare = market.loc[train_mask_np, "log_fare"].mean()
train_median_log_passengers = market.loc[train_mask_np, "log_passengers"].median()
train_median_hhi = market.loc[train_mask_np, "hhi"].median()

market["prev_log_fare"] = market["prev_log_fare"].fillna(train_mean_log_fare)
market["prev_log_passengers"] = market["prev_log_passengers"].fillna(train_median_log_passengers)
market["prev_hhi"] = market["prev_hhi"].fillna(train_median_hhi)


# ============================================================
# 14. Feature columns
# ============================================================

feature_cols = [
    "log_distance",
    "sin_q",
    "cos_q",
    "log_passengers",
    "log_n_records",
    "num_carriers",
    "hhi",
    "max_carrier_share",

    "log_origin_total_passengers",
    "log_dest_total_passengers",
    "log_origin_market_count",
    "log_dest_market_count",
    "log_origin_avg_distance",
    "log_dest_avg_distance",

    "origin_lat",
    "origin_lon",
    "dest_lat",
    "dest_lon",
    "route_lat_diff",
    "route_lon_diff",

    "prev_log_fare",
    "prev_log_passengers",
    "prev_hhi",
    "has_prev",
]

for col in feature_cols:
    market[col] = market[col].replace([np.inf, -np.inf], np.nan)
    market[col] = market[col].fillna(market.loc[train_mask_np, col].median())

print(f"\nFeature count: {len(feature_cols)}")
print("Feature columns:")
print(feature_cols)


# ============================================================
# 15. Build K2 market graph
# ============================================================

def add_edges_from_sorted_group(edges, group, k=2):
    ids = group["node_id"].to_numpy()

    if len(ids) <= 1:
        return

    for offset in range(1, k + 1):
        if len(ids) <= offset:
            break

        src = ids[:-offset]
        dst = ids[offset:]

        edges.extend(zip(src, dst))
        edges.extend(zip(dst, src))


def build_market_edges(market_df, k_neighbors=2):
    edges = []

    # Same route across time.
    for _, g in market_df.sort_values(["YEAR", "QUARTER"]).groupby(["ORIGIN", "DEST"], observed=True):
        ids = g["node_id"].to_numpy()
        if len(ids) > 1:
            edges.extend(zip(ids[:-1], ids[1:]))
            edges.extend(zip(ids[1:], ids[:-1]))

    # Same origin, same quarter, nearby route distance.
    for _, g in market_df.sort_values("distance_mean").groupby(["YEAR", "QUARTER", "ORIGIN"], observed=True):
        add_edges_from_sorted_group(edges, g, k=k_neighbors)

    # Same destination, same quarter, nearby route distance.
    for _, g in market_df.sort_values("distance_mean").groupby(["YEAR", "QUARTER", "DEST"], observed=True):
        add_edges_from_sorted_group(edges, g, k=k_neighbors)

    edge_df = pd.DataFrame(edges, columns=["src", "dst"]).drop_duplicates()
    edge_index = torch.tensor(edge_df[["src", "dst"]].values.T, dtype=torch.long).contiguous()

    return edge_index, edge_df


print("\nBuilding K2 market graph...")
edge_index, edge_df = build_market_edges(market, k_neighbors=K_NEIGHBORS)

print(f"Real graph edges: {edge_index.shape[1]:,}")

if RUN_RANDOM_EDGE_CONTROL:
    print("Building random edge control...")
    random_edge_index = build_random_edges(
        num_nodes=len(market),
        num_edges=edge_index.shape[1],
        seed=RANDOM_STATE + 999,
    )
    print(f"Random graph edges: {random_edge_index.shape[1]:,}")


# ============================================================
# 16. Prepare tensors
# ============================================================

scaler = StandardScaler()
X_train = market.loc[train_mask_np, feature_cols].values
scaler.fit(X_train)

X_all = scaler.transform(market[feature_cols].values).astype(np.float32)

y_all_log = market["log_fare"].values.astype(np.float32)

y_mean = y_all_log[train_mask_np].mean()
y_std = y_all_log[train_mask_np].std()

if y_std < 1e-8:
    y_std = 1.0

y_all_scaled = ((y_all_log - y_mean) / y_std).astype(np.float32)

print(f"\nTarget scaling: mean={y_mean:.4f}, std={y_std:.4f}")

x = torch.tensor(X_all, dtype=torch.float)
y_scaled = torch.tensor(y_all_scaled, dtype=torch.float)

train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

device = get_device()
print(f"Using device: {device}")


# ============================================================
# 17. Ridge baseline
# ============================================================

print("\n==============================")
print("1. Ridge baseline")
print("==============================")

ridge = Ridge(alpha=1.0)
ridge.fit(X_all[train_mask_np], y_all_log[train_mask_np])

ridge_val_pred_log = ridge.predict(X_all[val_mask_np])
ridge_test_pred_log = ridge.predict(X_all[test_mask_np])

print("\nRidge:")
ridge_val_metrics = eval_dollars(y_all_log[val_mask_np], ridge_val_pred_log, "Val")
ridge_test_metrics = eval_dollars(y_all_log[test_mask_np], ridge_test_pred_log, "Test")


# ============================================================
# 18. MLP baseline
# ============================================================

print("\n==============================")
print("2. MLP baseline")
print("==============================")

class MarketMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.15):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp():
    set_seed(RANDOM_STATE + 1)

    X_tensor = torch.tensor(X_all, dtype=torch.float).to(device)
    y_tensor = torch.tensor(y_all_scaled, dtype=torch.float).to(device)

    train_mask_t = torch.tensor(train_mask_np, dtype=torch.bool).to(device)
    val_mask_t = torch.tensor(val_mask_np, dtype=torch.bool).to(device)

    model = MarketMLP(
        in_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        dropout=MLP_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, MAX_EPOCHS_MLP + 1):
        model.train()

        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = F.mse_loss(pred[train_mask_t], y_tensor[train_mask_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_eval = model(X_tensor)
            val_loss = F.mse_loss(pred_eval[val_mask_t], y_tensor[val_mask_t]).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"MLP Epoch {epoch:03d} | train loss: {loss.item():.4f} | val loss: {val_loss:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"MLP early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).detach().cpu().numpy()

    pred_log = pred_scaled * y_std + y_mean

    return pred_scaled.astype(np.float32), pred_log.astype(np.float32)


mlp_pred_scaled, mlp_pred_log = train_mlp()

print("\nMLP:")
mlp_val_metrics = eval_dollars(y_all_log[val_mask_np], mlp_pred_log[val_mask_np], "Val")
mlp_test_metrics = eval_dollars(y_all_log[test_mask_np], mlp_pred_log[test_mask_np], "Test")


# ============================================================
# 19. MLP-residual GraphCorrection GNN
# ============================================================

print("\n==============================")
print("3. MLP-residual K2 GraphCorrection GNN")
print("==============================")

class ResidualGraphCorrectionGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.15):
        super().__init__()

        self.input = nn.Linear(in_dim, hidden_dim)

        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Node-wise gate: model can use graph correction more on hard/sparse markets.
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        # Conservative global multiplier.
        self.global_gate_logit = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x, edge_index):
        h0 = F.relu(self.input(x))

        h1 = self.conv1(h0, edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1 + h0)

        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2 + h1)

        residual = self.residual_head(h2).squeeze(-1)

        node_gate = torch.sigmoid(self.gate_head(h0)).squeeze(-1)
        global_gate = torch.sigmoid(self.global_gate_logit)

        return global_gate * node_gate * residual


def train_residual_gnn(edge_index_to_use, label="Real-GNN"):
    set_seed(RANDOM_STATE + 2)

    data = Data(
        x=x,
        edge_index=edge_index_to_use,
        y=y_scaled,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    ).to(device)

    mlp_pred_t = torch.tensor(mlp_pred_scaled, dtype=torch.float).to(device)

    model = ResidualGraphCorrectionGNN(
        in_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        dropout=GNN_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GNN_LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = float("inf")
    best_state = None
    best_gate = None
    patience_ctr = 0

    print(f"\nTraining {label} | edges: {data.edge_index.shape[1]:,}")

    for epoch in range(1, MAX_EPOCHS_GNN + 1):
        model.train()

        optimizer.zero_grad()

        residual_pred = model(data.x, data.edge_index)
        final_pred = mlp_pred_t + residual_pred

        loss = F.mse_loss(final_pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            residual_eval = model(data.x, data.edge_index)
            final_eval = mlp_pred_t + residual_eval
            val_loss = F.mse_loss(final_eval[data.val_mask], data.y[data.val_mask]).item()
            global_gate = torch.sigmoid(model.global_gate_logit).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_gate = global_gate
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(
                f"{label} Epoch {epoch:03d} | "
                f"train loss: {loss.item():.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"global_gate: {global_gate:.4f}"
            )

        if patience_ctr >= PATIENCE:
            print(f"{label} early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        residual_pred = model(data.x, data.edge_index).detach().cpu().numpy()

    final_pred_scaled = mlp_pred_scaled + residual_pred
    final_pred_log = final_pred_scaled * y_std + y_mean

    print(f"{label} best global gate: {best_gate:.4f}")

    val_metrics = eval_dollars(y_all_log[val_mask_np], final_pred_log[val_mask_np], "Val")
    test_metrics = eval_dollars(y_all_log[test_mask_np], final_pred_log[test_mask_np], "Test")

    return val_metrics, test_metrics, best_gate


print("\nReal K2 graph:")
gnn_val_metrics, gnn_test_metrics, gnn_gate = train_residual_gnn(edge_index, label="Real-K2-GNN")

if RUN_RANDOM_EDGE_CONTROL:
    print("\nRandom edge control:")
    rand_val_metrics, rand_test_metrics, rand_gate = train_residual_gnn(
        random_edge_index,
        label="Random-GNN",
    )


# ============================================================
# 20. Final comparison
# ============================================================

print("\n================================================================================")
print(f"Final Model Comparison — Market-as-node | split={SPLIT_MODE}")
print("================================================================================")
print(
    f"{'Model':<28} {'Split':<8} "
    f"{'RMSE($)':>10} {'MAE($)':>10} {'MedAE($)':>10} {'R2(log)':>10} {'Edges':>10} {'Gate':>8}"
)
print("--------------------------------------------------------------------------------")

summary_rows = [
    ("Ridge", "Val", ridge_val_metrics, 0, np.nan),
    ("Ridge", "Test", ridge_test_metrics, 0, np.nan),
    ("MLP", "Val", mlp_val_metrics, 0, np.nan),
    ("MLP", "Test", mlp_test_metrics, 0, np.nan),
    ("MLP-residual K2 GNN", "Val", gnn_val_metrics, edge_index.shape[1], gnn_gate),
    ("MLP-residual K2 GNN", "Test", gnn_test_metrics, edge_index.shape[1], gnn_gate),
]

if RUN_RANDOM_EDGE_CONTROL:
    summary_rows.extend([
        ("Random-edge GNN", "Val", rand_val_metrics, random_edge_index.shape[1], rand_gate),
        ("Random-edge GNN", "Test", rand_test_metrics, random_edge_index.shape[1], rand_gate),
    ])

for model_name, split_name, metrics, num_edges, gate in summary_rows:
    rmse, mae, medae, r2 = metrics
    gate_str = "" if np.isnan(gate) else f"{gate:.3f}"

    print(
        f"{model_name:<28} {split_name:<8} "
        f"{rmse:>10.2f} {mae:>10.2f} {medae:>10.2f} {r2:>10.3f} "
        f"{num_edges:>10} {gate_str:>8}"
    )

print("================================================================================")


# ============================================================
# 21. Save outputs
# ============================================================

results = []

for model_name, split_name, metrics, num_edges, gate in summary_rows:
    rmse, mae, medae, r2 = metrics

    results.append({
        "model": model_name,
        "split": split_name,
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "r2_log": r2,
        "num_edges": num_edges,
        "gate": gate,
        "split_mode": SPLIT_MODE,
        "row_sample_frac": ROW_SAMPLE_FRAC,
        "max_train_nodes": MAX_TRAIN_NODES,
        "max_val_nodes": MAX_VAL_NODES,
        "max_test_nodes": MAX_TEST_NODES,
        "total_market_nodes": len(market),
    })

results_path = f"market_node_final_results_{SPLIT_MODE}.csv"
nodes_path = f"market_nodes_final_{SPLIT_MODE}.csv"
edges_path = f"market_edges_k2_{SPLIT_MODE}.csv"

pd.DataFrame(results).to_csv(results_path, index=False)
market.to_csv(nodes_path, index=False)
edge_df.to_csv(edges_path, index=False)

print(f"\nSaved results to: {results_path}")
print(f"Saved market nodes to: {nodes_path}")
print(f"Saved K2 edges to: {edges_path}")