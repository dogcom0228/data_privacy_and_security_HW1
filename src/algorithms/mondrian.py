from typing import List, Dict, Any, Tuple, Optional

Record = Dict[str, Any]


def mondrian_k_anonymity(
    data: List[Record],
    qi_names: List[str],
    k: int,
    is_categorical: Dict[str, bool],
) -> List[Record]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not data:
        return []
    if len(data) < k:
        raise ValueError(
            f"Dataset size ({len(data)}) is smaller than k ({k}); "
            "cannot achieve k-anonymity"
        )

    global_stats = _compute_global_stats(data, qi_names, is_categorical)

    partitions: List[List[Record]] = []
    _mondrian_partition(
        records=data,
        qi_names=qi_names,
        k=k,
        is_categorical=is_categorical,
        global_stats=global_stats,
        out_partitions=partitions,
    )

    anonymized: List[Record] = []
    for part in partitions:
        anonymized.extend(
            _generalize_partition(part, qi_names, is_categorical)
        )
    return anonymized


# ------------------------------------------------------------------ helpers


def _compute_global_stats(
    data: List[Record],
    qi_names: List[str],
    is_categorical: Dict[str, bool],
) -> dict:
    stats: dict = {}
    for qi in qi_names:
        vals = [r.get(qi) for r in data if r.get(qi) is not None]
        if not is_categorical.get(qi, False):
            gmin = min(vals) if vals else 0.0
            gmax = max(vals) if vals else 0.0
            stats[qi] = {
                "type": "numeric",
                "global_min": gmin,
                "global_max": gmax,
            }
        else:
            cats = sorted(set(vals))
            stats[qi] = {
                "type": "categorical",
                "categories": cats,
                "cat2idx": {c: i for i, c in enumerate(cats)},
            }
    return stats


def _val_to_number(
    val: Any,
    qi: str,
    is_categorical: Dict[str, bool],
    global_stats: dict,
) -> float:
    if val is None:
        return -float("inf")
    if not is_categorical.get(qi, False):
        return float(val)
    return float(global_stats[qi]["cat2idx"].get(val, -1))


def _normalized_range(
    records: List[Record],
    qi: str,
    is_categorical: Dict[str, bool],
    global_stats: dict,
) -> float:
    vals = [r.get(qi) for r in records if r.get(qi) is not None]
    if not vals:
        return 0.0

    if not is_categorical.get(qi, False):
        gmin = global_stats[qi]["global_min"]
        gmax = global_stats[qi]["global_max"]
        if gmax == gmin:
            return 0.0
        return (max(vals) - min(vals)) / (gmax - gmin)

    cat2idx = global_stats[qi]["cat2idx"]
    idxs = [cat2idx[v] for v in vals]
    total = len(global_stats[qi]["categories"])
    if total <= 1:
        return 0.0
    return (max(idxs) - min(idxs)) / (total - 1)


def _ranked_dimensions(
    records: List[Record],
    qi_names: List[str],
    is_categorical: Dict[str, bool],
    global_stats: dict,
) -> List[str]:
    """Return QI names sorted by descending normalized range (skip range==0)."""
    pairs = []
    for qi in qi_names:
        nr = _normalized_range(records, qi, is_categorical, global_stats)
        if nr > 0.0:
            pairs.append((nr, qi))
    pairs.sort(reverse=True)
    return [qi for _, qi in pairs]


# ------------------------------------------------------------------ split


def _split_on_dimension(
    records: List[Record],
    dim: str,
    is_categorical: Dict[str, bool],
    global_stats: dict,
) -> Tuple[Optional[List[Record]], Optional[List[Record]]]:
    """Split records by median value on *dim*, keeping identical values together."""
    keyed = [
        (_val_to_number(r.get(dim), dim, is_categorical, global_stats), r)
        for r in records
    ]
    keyed.sort(key=lambda x: x[0])

    median_val = keyed[len(keyed) // 2][0]

    # Attempt 1: left <= median, right > median
    left = [r for v, r in keyed if v <= median_val]
    right = [r for v, r in keyed if v > median_val]

    if not right:
        # Attempt 2: left < median, right >= median
        left = [r for v, r in keyed if v < median_val]
        right = [r for v, r in keyed if v >= median_val]

    if not left or not right:
        return None, None

    return left, right


# ------------------------------------------------------------------ recurse


def _mondrian_partition(
    records: List[Record],
    qi_names: List[str],
    k: int,
    is_categorical: Dict[str, bool],
    global_stats: dict,
    out_partitions: List[List[Record]],
) -> None:
    if len(records) < 2 * k:
        out_partitions.append(records)
        return

    # Try each dimension in decreasing normalized-range order
    for dim in _ranked_dimensions(records, qi_names, is_categorical, global_stats):
        left, right = _split_on_dimension(records, dim, is_categorical, global_stats)
        if left is not None and right is not None                 and len(left) >= k and len(right) >= k:
            _mondrian_partition(left, qi_names, k, is_categorical, global_stats, out_partitions)
            _mondrian_partition(right, qi_names, k, is_categorical, global_stats, out_partitions)
            return

    # No valid split found on any dimension
    out_partitions.append(records)


# ------------------------------------------------------------------ generalize


def _generalize_partition(
    part: List[Record],
    qi_names: List[str],
    is_categorical: Dict[str, bool],
) -> List[Record]:
    if not part:
        return []

    gen_values: Dict[str, Any] = {}
    for qi in qi_names:
        vals = [r.get(qi) for r in part if r.get(qi) is not None]
        if not vals:
            gen_values[qi] = None
            continue
        if not is_categorical.get(qi, False):
            vmin, vmax = min(vals), max(vals)
            gen_values[qi] = vmin if vmin == vmax else (vmin, vmax)
        else:
            uniq = sorted(set(vals))
            gen_values[qi] = uniq[0] if len(uniq) == 1 else tuple(uniq)

    result: List[Record] = []
    for r in part:
        new_r = r.copy()
        for qi in qi_names:
            # Preserve None so missing values are not disguised as real ranges
            if r.get(qi) is None:
                new_r[qi] = None
            else:
                new_r[qi] = gen_values[qi]
        result.append(new_r)
    return result
