"""Filter WLASL annotations to the top-N glosses by sample count.

Requisitos: 6.2
"""

from __future__ import annotations


def filter_top_n(
    annotations: list[dict],
    n: int = 50,
) -> tuple[list[dict], list[str]]:
    """Select the top-n glosses by number of instances.

    Args:
        annotations: List of dicts, each with at least
            ``{"gloss": str, "instances": list}``.
        n: Number of top glosses to keep.

    Returns:
        A tuple ``(filtered_annotations, label_list)`` where:
        - ``filtered_annotations`` contains only entries whose gloss is in the
          top-n set.
        - ``label_list`` is a sorted list of the selected gloss strings.
    """
    # Count samples per gloss
    counts: dict[str, int] = {
        entry["gloss"]: len(entry["instances"]) for entry in annotations
    }

    # Select top-n glosses by count (stable: ties broken by gloss name via sort)
    top_glosses: set[str] = {
        gloss
        for gloss, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    }

    filtered_annotations = [a for a in annotations if a["gloss"] in top_glosses]
    label_list = sorted(top_glosses)

    return filtered_annotations, label_list
