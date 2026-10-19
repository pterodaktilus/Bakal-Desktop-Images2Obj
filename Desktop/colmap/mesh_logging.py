import time

import open3d as o3d


LOG_PREFIX = "[PointCloud2Obj]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def elapsed_since(started_at: float) -> str:
    seconds = time.perf_counter() - started_at
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {seconds:.1f}s"


def log_mesh(
    label: str,
    mesh: o3d.geometry.TriangleMesh,
    started_at: float | None = None,
    topology: bool = False,
) -> tuple[bool, bool, bool] | None:
    details = [
        f"vertices={len(mesh.vertices):,}",
        f"triangles={len(mesh.triangles):,}",
    ]
    if started_at is not None:
        details.append(f"time={elapsed_since(started_at)}")
    if not topology:
        log(f"{label} | {' | '.join(details)}")
        return None

    state = (
        mesh.is_edge_manifold(True),
        mesh.is_vertex_manifold(),
        mesh.is_orientable(),
    )
    details.extend([
        f"edge_manifold={'yes' if state[0] else 'no'}",
        f"vertex_manifold={'yes' if state[1] else 'no'}",
        f"orientable={'yes' if state[2] else 'no'}",
    ])
    log(f"{label} | {' | '.join(details)}")
    return state
