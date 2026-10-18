import time
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d

from ColmapsReconstruction import ColmapsReconstruction


Mesher = Literal["poisson", "delaunay", "advancing-front", "open3d"]
Simplifier = Literal["colmap", "open3d"]
LOG_PREFIX = "[ColmapMeshPipeline]"


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
    started_at: float,
) -> None:
    log(
        f"{label} | vertices={len(mesh.vertices):,}"
        f" | triangles={len(mesh.triangles):,}"
        f" | edge_manifold={'yes' if mesh.is_edge_manifold(True) else 'no'}"
        f" | vertex_manifold={'yes' if mesh.is_vertex_manifold() else 'no'}"
        f" | orientable={'yes' if mesh.is_orientable() else 'no'}"
        f" | time={elapsed_since(started_at)}"
    )


class ColmapMeshPipeline(ColmapsReconstruction):
    """Create, simplify, clean, texture, and export a dense COLMAP mesh."""

    MESHERS = ("poisson", "delaunay", "advancing-front", "open3d")
    SIMPLIFIERS = ("colmap", "open3d")

    @staticmethod
    def _require_nonempty_file(path: Path, description: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"{description} is empty: {path}")

    @staticmethod
    def _derived_path(path: Path, suffix: str) -> Path:
        return path.with_name(f"{path.stem}-{suffix}{path.suffix}")

    def _create_open3d_surface(self) -> Path:
        """Create a Poisson surface directly from COLMAP's fused point cloud."""
        self._require_nonempty_file(
            self.fused_point_cloud,
            "Fused point cloud",
        )
        if not 0 <= self.poisson_density_quantile < 1:
            raise ValueError(
                "poisson_density_quantile must be at least 0 and less than 1"
            )

        started_at = time.perf_counter()
        log(f"Open3D point-cloud loading started | input={self.fused_point_cloud}")
        cloud = o3d.io.read_point_cloud(
            str(self.fused_point_cloud),
            remove_nan_points=True,
            remove_infinite_points=True,
            print_progress=True,
        )
        if not cloud.has_points():
            raise RuntimeError(
                f"Open3D could not read point-cloud data: {self.fused_point_cloud}"
            )
        if not cloud.has_normals():
            raise RuntimeError(
                "Open3D Poisson reconstruction requires point normals, but the "
                "fused point cloud does not contain them."
            )
        log(
            f"Open3D point-cloud loading complete | points={len(cloud.points):,}"
            f" | colors={'yes' if cloud.has_colors() else 'no'}"
            f" | time={elapsed_since(started_at)}"
        )

        started_at = time.perf_counter()
        log("Open3D Poisson meshing started | depth=10")
        mesh, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud,
                depth=11,
            )
        )
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise RuntimeError("Open3D Poisson reconstruction produced an empty mesh.")
        log_mesh("Open3D Poisson meshing complete", mesh, started_at)

        density_values = np.asarray(densities)
        if density_values.size != len(mesh.vertices):
            raise RuntimeError(
                "Open3D Poisson returned a density count that does not match "
                "the mesh vertices."
            )
        density_threshold = np.quantile(
            density_values,
            self.poisson_density_quantile,
        )
        low_density_vertices = density_values < density_threshold
        log(
            "Open3D density filtering started"
            f" | quantile={self.poisson_density_quantile:.1%}"
            f" | vertices_to_remove={np.count_nonzero(low_density_vertices):,}"
        )
        mesh.remove_vertices_by_mask(low_density_vertices)
        mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise RuntimeError(
                "Open3D filtering removed all usable mesh geometry."
            )
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        output_path = self.open3d_mesh
        if not o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            print_progress=True,
        ):
            raise RuntimeError(f"Open3D failed to write mesh: {output_path}")
        self._require_nonempty_file(output_path, "Open3D mesh")
        log(f"Open3D surface creation complete | output={output_path}")
        return output_path

    def create_surface(self, mesher: Mesher) -> Path:
        """Create a surface using a COLMAP mesher or Open3D Poisson."""
        if mesher not in self.MESHERS:
            raise ValueError(
                f"Unknown mesher '{mesher}'. Choose one of: "
                f"{', '.join(self.MESHERS)}"
            )

        if mesher == "open3d":
            return self._create_open3d_surface()
        if mesher == "poisson":
            self._require_nonempty_file(
                self.fused_point_cloud,
                "Fused point cloud",
            )
            output_path = self.poisson_mesh
            log(
                "COLMAP Poisson meshing started"
                f" | trim={self.poisson_trim:g}"
                f" | input={self.fused_point_cloud}"
            )
            self._run_colmap(
                "poisson_mesher",
                "--input_path",
                self.fused_point_cloud,
                "--output_path",
                output_path,
                "--PoissonMeshing.trim",
                self.poisson_trim,
            )
        elif mesher == "delaunay":
            if not self.dense.is_dir():
                raise FileNotFoundError(
                    f"Dense COLMAP workspace does not exist: {self.dense}"
                )
            output_path = self.delaunay_mesh
            log(f"COLMAP Delaunay meshing started | workspace={self.dense}")
            self._run_colmap(
                "delaunay_mesher",
                "--input_path",
                self.dense,
                "--input_type",
                "dense",
                "--output_path",
                output_path,
            )
        else:
            if not self.dense.is_dir():
                raise FileNotFoundError(
                    f"Dense COLMAP workspace does not exist: {self.dense}"
                )
            output_path = self.advancing_front_mesh
            log(
                "COLMAP advancing-front meshing started"
                f" | workspace={self.dense}"
            )
            self._run_colmap(
                "advancing_front_mesher",
                "--input_path",
                self.dense,
                "--output_path",
                output_path,
            )

        self._require_nonempty_file(output_path, "COLMAP mesh")
        log(
            f"COLMAP meshing complete | method={mesher}"
            f" | output={output_path}"
        )
        return output_path

    def simplify_mesh(
        self,
        input_path: Path,
        simplifier: Simplifier = "colmap",
    ) -> Path:
        """Simplify a mesh using the selected quadric-error simplifier."""
        self._require_nonempty_file(input_path, "Input mesh")
        if not 0 < self.mesh_target_face_ratio <= 1:
            raise ValueError(
                "mesh_target_face_ratio must be greater than 0 and at most 1"
            )
        if simplifier not in self.SIMPLIFIERS:
            raise ValueError(
                f"Unknown simplifier '{simplifier}'. Choose one of: "
                f"{', '.join(self.SIMPLIFIERS)}"
            )

        output_path = self._derived_path(input_path, "simplified")
        if simplifier == "open3d":
            started_at = time.perf_counter()
            mesh = o3d.io.read_triangle_mesh(
                str(input_path),
                enable_post_processing=False,
                print_progress=True,
            )
            if not mesh.has_vertices() or not mesh.has_triangles():
                raise RuntimeError(
                    f"Open3D could not read mesh geometry: {input_path}"
                )

            source_triangle_count = len(mesh.triangles)
            target_triangle_count = max(
                1,
                int(source_triangle_count * self.mesh_target_face_ratio),
            )
            log(
                "Open3D mesh simplification started"
                f" | source_triangles={source_triangle_count:,}"
                f" | target_triangles={target_triangle_count:,}"
                f" | retained={self.mesh_target_face_ratio:.1%}"
            )
            simplified_mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangle_count,
            )
            if (
                not simplified_mesh.has_vertices()
                or not simplified_mesh.has_triangles()
            ):
                raise RuntimeError(
                    "Open3D mesh simplification produced an empty mesh."
                )
            simplified_mesh.compute_triangle_normals()
            simplified_mesh.compute_vertex_normals()
            log_mesh(
                "Open3D mesh simplification complete",
                simplified_mesh,
                started_at,
            )
            if not o3d.io.write_triangle_mesh(
                str(output_path),
                simplified_mesh,
                write_ascii=False,
                compressed=False,
                write_vertex_normals=True,
                write_vertex_colors=True,
                print_progress=True,
            ):
                raise RuntimeError(
                    f"Open3D failed to write simplified mesh: {output_path}"
                )
            self._require_nonempty_file(output_path, "Simplified mesh")
            return output_path

        log(
            "COLMAP mesh simplification started"
            f" | retained={self.mesh_target_face_ratio:.1%}"
        )
        self._run_colmap(
            "mesh_simplifier",
            "--input_path",
            input_path,
            "--output_path",
            output_path,
            "--MeshSimplification.target_face_ratio",
            self.mesh_target_face_ratio,
        )
        self._require_nonempty_file(output_path, "Simplified mesh")
        log(f"COLMAP mesh simplification complete | output={output_path}")
        return output_path

    def clean_mesh(self, input_path: Path) -> tuple[Path, o3d.geometry.TriangleMesh]:
        """Use Open3D for cleanup operations without a COLMAP equivalent."""
        self._require_nonempty_file(input_path, "Input mesh")
        started_at = time.perf_counter()
        log(f"Open3D mesh cleanup started | input={input_path}")

        mesh = o3d.io.read_triangle_mesh(
            str(input_path),
            enable_post_processing=False,
            print_progress=True,
        )
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise RuntimeError(f"Open3D could not read mesh geometry: {input_path}")

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise RuntimeError("Open3D cleanup removed all usable mesh geometry.")

        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        log_mesh("Open3D mesh cleanup complete", mesh, started_at)

        output_path = self._derived_path(input_path, "cleaned")
        if not o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            print_progress=True,
        ):
            raise RuntimeError(f"Open3D failed to write cleaned mesh: {output_path}")
        self._require_nonempty_file(output_path, "Cleaned mesh")
        return output_path, mesh

    def texture_mesh(self, input_path: Path) -> tuple[Path, Path]:
        """Create per-face UVs and an image atlas using COLMAP camera views."""
        self._require_nonempty_file(input_path, "Input mesh")
        sparse_model = self.dense / "sparse"
        if not sparse_model.is_dir():
            raise FileNotFoundError(
                "The dense workspace has no undistorted sparse model: "
                f"{sparse_model}"
            )

        output_directory = self.dense / f"{input_path.stem}-textured"
        output_directory.mkdir(parents=True, exist_ok=True)
        log(
            "COLMAP multi-view texturing started"
            f" | mesh={input_path}"
            f" | output={output_directory}"
        )
        self._run_colmap(
            "mesh_texturer",
            "--workspace_path",
            self.dense,
            "--input_path",
            input_path,
            "--output_path",
            output_directory,
            "--output_type",
            "BIN",
        )

        textured_mesh = output_directory / "mesh.ply"
        texture_atlas = output_directory / "texture.png"
        self._require_nonempty_file(textured_mesh, "COLMAP textured mesh")
        self._require_nonempty_file(texture_atlas, "COLMAP texture atlas")
        log(
            "COLMAP multi-view texturing complete"
            f" | mesh={textured_mesh}"
            f" | atlas={texture_atlas}"
        )
        return textured_mesh, texture_atlas

    def export_vertex_color_glb(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_path: Path,
    ) -> Path:
        """Use Open3D to create the GLB output that COLMAP cannot export."""
        if not mesh.has_vertex_colors():
            raise RuntimeError(
                "Open3D GLB export requires vertex colors, but the COLMAP mesh "
                "does not contain them. The COLMAP PLY and PNG texture outputs "
                "are still usable."
            )
        if self.texture_size < 1:
            raise ValueError("texture_size must be at least 1")

        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        log(
            "Open3D UV atlas started"
            f" | size={self.texture_size}x{self.texture_size}"
        )
        started_at = time.perf_counter()
        stretch, charts, partitions = tensor_mesh.compute_uvatlas(
            size=self.texture_size,
            gutter=self.texture_gutter,
        )
        log(
            f"Open3D UV atlas complete | charts={charts:,}"
            f" | partitions={partitions:,}"
            f" | max_stretch={stretch:.4f}"
            f" | time={elapsed_since(started_at)}"
        )

        tensor_mesh.vertex["albedo"] = tensor_mesh.vertex.colors
        tensor_mesh.material.set_default_properties()
        textures = tensor_mesh.bake_vertex_attr_textures(
            self.texture_size,
            {"albedo"},
            margin=self.texture_gutter,
            update_material=False,
        )
        albedo = (textures["albedo"].clip(0, 1) * 255).to(
            o3d.core.Dtype.UInt8
        )
        tensor_mesh.material.texture_maps["albedo"] = o3d.t.geometry.Image(albedo)

        log(f"Open3D GLB export started | output={output_path}")
        if not o3d.t.io.write_triangle_mesh(str(output_path), tensor_mesh):
            raise RuntimeError(f"Open3D failed to export GLB: {output_path}")
        self._require_nonempty_file(output_path, "Open3D GLB")
        log(
            f"Open3D GLB export complete | output={output_path}"
            f" | size={output_path.stat().st_size / (1024 * 1024):.2f} MiB"
        )
        return output_path

    def run(
        self,
        mesher: Mesher = "poisson",
        export_glb: bool = True,
        simplifier: Simplifier | None = None,
    ) -> dict[str, Path]:
        """Run the hybrid COLMAP/Open3D mesh pipeline."""
        started_at = time.perf_counter()
        if simplifier is None:
            simplifier = "open3d" if mesher == "open3d" else "colmap"
        surface_mesh = self.create_surface(mesher)
        simplified_mesh = self.simplify_mesh(surface_mesh, simplifier)
        cleaned_mesh, open3d_mesh = self.clean_mesh(simplified_mesh)
        textured_mesh, texture_atlas = self.texture_mesh(cleaned_mesh)

        outputs = {
            "surface_mesh": surface_mesh,
            "simplified_mesh": simplified_mesh,
            "cleaned_mesh": cleaned_mesh,
            "textured_mesh": textured_mesh,
            "texture_atlas": texture_atlas,
        }
        if export_glb:
            outputs["glb"] = self.export_vertex_color_glb(
                open3d_mesh,
                self.textured_mesh,
            )

        log(f"Mesh pipeline complete | time={elapsed_since(started_at)}")
        return outputs
