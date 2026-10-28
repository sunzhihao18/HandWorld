import numpy as np
from scipy.signal import savgol_filter


def remove_isolated(temporal_vertices, min_seg_len=2):
    result = temporal_vertices.copy()
    start = None
    for i, v in enumerate(temporal_vertices + [None]):
        if v is not None:
            if start is None:
                start = i
        elif start is not None:
            seg_len = i - start
            if seg_len < min_seg_len:
                for j in range(start, i):
                    result[j] = None
            start = None
    return result


def filter_outliers(temporal_vertices, window_size=9, thresh_scale=2):
    num_frames = len(temporal_vertices)
    centers = np.full((num_frames, 3), np.nan)
    for i, v in enumerate(temporal_vertices):
        if v is not None:
            centers[i] = np.mean(v, axis=0)
    result = temporal_vertices.copy()
    half_w = window_size // 2
    for i in range(num_frames):
        _thresh_scale = thresh_scale
        if np.any(np.isnan(centers[i])):
            continue
        start = i - half_w
        if start < 0:
            start = 0
            _thresh_scale = 1
        end = min(num_frames, i + half_w + 1)
        window_vals = centers[start:end]
        valid = ~np.isnan(window_vals[:, 0])
        if valid.sum() < 3:
            continue
        local_mean = np.nanmean(window_vals[valid], axis=0)
        local_std = np.nanstd(window_vals[valid], axis=0) + 1e-08
        dist = np.linalg.norm(centers[i] - local_mean)
        threshold = _thresh_scale * np.mean(local_std)
        if dist > threshold:
            result[i] = None
    return result


def interpolate(temporal_vertices, max_gap=5):
    result = temporal_vertices.copy()
    valid_indices = [i for i, v in enumerate(temporal_vertices) if v is not None]
    if len(valid_indices) < 2:
        return temporal_vertices
    for i in range(len(valid_indices) - 1):
        i0, i1 = (valid_indices[i], valid_indices[i + 1])
        gap = i1 - i0 - 1
        if 0 < gap <= max_gap:
            v0 = temporal_vertices[i0]
            v1 = temporal_vertices[i1]
            for j in range(1, gap + 1):
                alpha = j / (gap + 1)
                result[i0 + j] = (1 - alpha) * v0 + alpha * v1
    return result


def smooth_savgol(temporal_vertices, window_size=5, poly_order=2):

    def smooth_seg(seg):
        seg = np.stack(seg, axis=0)
        seg_s = savgol_filter(
            x=seg,
            window_length=min(window_size, (len(seg) - 1) // 2 * 2 + 1),
            polyorder=min(poly_order, len(seg) - 1),
            axis=0,
            mode="interp",
        )
        return seg_s

    result = temporal_vertices.copy()
    seg = []
    start_idx = None
    for i, v in enumerate(temporal_vertices + [None]):
        if v is not None:
            if start_idx is None:
                start_idx = i
            seg.append(v)
        elif seg:
            if len(seg) < 3:
                continue
            seg_s = smooth_seg(seg)
            for k, v_s in enumerate(seg_s):
                result[start_idx + k] = v_s
            seg = []
            start_idx = None
    return result
