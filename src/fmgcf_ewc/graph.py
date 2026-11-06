import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def knn_adjacency(regions, k):
    codes = list(regions["region"])
    lat = regions["lat"].to_numpy()
    lon = regions["lon"].to_numpy()
    distance = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    n = len(codes)
    adjacency = np.zeros((n, n))
    for i in range(n):
        order = [j for j in np.argsort(distance[i]) if j != i][: min(k, n - 1)]
        adjacency[i, order] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(n)
    normalised = adjacency / adjacency.sum(axis=1, keepdims=True)
    return codes, normalised, distance


def propagate(panel, regions, columns, k, hops=1):
    codes, normalised, _ = knn_adjacency(regions, k)
    operator = np.linalg.matrix_power(normalised, hops)
    position = {code: i for i, code in enumerate(codes)}
    out = panel.copy()
    for column in columns:
        out[f"{column}_neighbours"] = np.nan
    for (year, week), block in panel.groupby(["iso_year", "iso_week"]):
        vector_index = block["region"].map(position).to_numpy()
        for column in columns:
            values = np.full(len(codes), np.nan)
            values[vector_index] = block[column].to_numpy()
            mask = ~np.isnan(values)
            filled = np.where(mask, values, 0.0)
            weight = operator @ mask.astype(float)
            smoothed = np.divide(operator @ filled, weight, out=np.full(len(codes), np.nan), where=weight > 0)
            out.loc[block.index, f"{column}_neighbours"] = smoothed[vector_index]
    return out


def edge_list(regions, k):
    codes, normalised, distance = knn_adjacency(regions, k)
    rows = []
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            if i < j and normalised[i, j] > 0:
                rows.append({"region_a": a, "region_b": b, "distance_km": float(distance[i, j]), "cross_border": a[:2] != b[:2]})
    return pd.DataFrame(rows)
