"""Raw dataset loading helpers for PathSnap."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "Student folder" / "Student folder"


def list_trips(partition_dir: Path) -> List[Path]:
    """Return sorted trip directories inside a partition directory."""
    partition_dir = Path(partition_dir)
    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {partition_dir}")
    return sorted([p for p in partition_dir.iterdir() if p.is_dir()])


def load_trip(trip_dir: Path, load_route: bool = True) -> Dict[str, object]:
    """Load .track, .nodes, .arcs, and optionally .route for a single trip."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64},
    )

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int64, "to_node": np.int64},
    )

    route: Optional[np.ndarray] = None
    route_path = trip_dir / f"{trip_id}.route"
    if load_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64, ndmin=1)

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }
