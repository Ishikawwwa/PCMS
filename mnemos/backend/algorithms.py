"""All graph decisions use original vectors; projection is a separate operation."""
import itertools

import numpy as np


def normalized(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError("Expected finite nonempty embedding matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero embedding vectors cannot be normalized")
    return array / norms


def discover(ids, vectors, settings, source_sets):
    if len(ids) < settings.min_size:
        return []
    matrix = normalized(vectors) @ normalized(vectors).T
    neighborhoods = {i: sorted((j for j in range(len(ids)) if j != i), key=lambda j: (-matrix[i, j], ids[j]))[:settings.top_k] for i in range(len(ids))}
    seeds = sorted({tuple(sorted((i, j))) for i in neighborhoods for j in neighborhoods[i] if matrix[i, j] >= max(settings.threshold, settings.min_pairwise)}, key=lambda p: (-matrix[p], p))
    candidates = {}
    for a, b in seeds:
        group = [a, b]
        pool = set(neighborhoods[a]) | set(neighborhoods[b])
        for candidate in sorted(pool - set(group), key=lambda j: (-min(matrix[j, i] for i in group), ids[j])):
            if len(group) >= settings.max_size:
                break
            if all(matrix[candidate, i] >= settings.min_pairwise for i in group) and max(matrix[candidate, i] for i in group) >= settings.threshold:
                group.append(candidate)
        if len(group) < settings.min_size:
            continue
        group.sort()
        pairs = [float(matrix[i, j]) for i, j in itertools.combinations(group, 2)]
        members = [ids[i] for i in group]
        diversity = len(set().union(*(source_sets[m] for m in members)))
        candidates[tuple(members)] = {"member_ids": members, "pairwise_similarity": matrix[np.ix_(group, group)].tolist(), "mean_pairwise_cosine": float(np.mean(pairs)), "min_pairwise_cosine": min(pairs), "cohesion": float(np.mean(pairs)), "evidence_diversity": diversity, "ranking_score": float(np.mean(pairs)) + .05 * min(diversity, 5)}
    result = []
    for candidate in sorted(candidates.values(), key=lambda c: (-c["ranking_score"], c["member_ids"])):
        members = set(candidate["member_ids"])
        overlaps = [len(members & set(c["member_ids"])) / len(members | set(c["member_ids"])) for c in result]
        if any(overlap >= settings.jaccard for overlap in overlaps):
            continue
        candidate["max_jaccard_overlap"] = max(overlaps, default=0)
        result.append(candidate)
    return result


def projection(ids, vectors):
    x = normalized(vectors)
    center = x.mean(axis=0)
    _, singular, components = np.linalg.svd(x - center, full_matrices=False)
    basis = components[:3]
    # Fix the SVD sign ambiguity for reproducible display coordinates.
    for row in basis:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    coordinates = (x - center) @ basis.T
    coordinates = np.pad(coordinates, ((0, 0), (0, 3 - coordinates.shape[1])))
    total = float((singular ** 2).sum())
    variance = float((singular[:3] ** 2).sum() / total) if total else 0
    return {"coordinates": dict(zip(ids, coordinates.tolist())), "center": center.tolist(), "components": basis.tolist(), "explained_variance": variance, "warning": "PCA distances approximate the original embedding space. Relation membership uses original cosine only."}
