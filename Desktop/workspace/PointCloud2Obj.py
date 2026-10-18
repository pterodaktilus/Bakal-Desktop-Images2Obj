import open3d as o3d
import numpy as np
import pymeshlab
import time
from config import ColmapsConfig
from mesh_logging import elapsed_since, log, log_mesh


class MVSOpen3d(ColmapsConfig):
    def __init__(self) -> None:
        super().__init__()
        self.cloud = None

    def LoadPoinCloud(self) -> None:
        started_at = time.perf_counter()
        log(f"Loading point cloud | path={self.fused_point_cloud}")
        if not self.fused_point_cloud.is_file():
            raise FileNotFoundError(
                f"Point-cloud file does not exist: {self.fused_point_cloud}")

        if self.fused_point_cloud.stat().st_size == 0:
            raise RuntimeError(
                f"Point-cloud file is empty: {self.fused_point_cloud}")

        self.cloud = o3d.io.read_point_cloud(
            str(self.fused_point_cloud),
            remove_nan_points=True,
            remove_infinite_points=True,
            print_progress=True,)

        if not self.cloud.has_points():
            raise RuntimeError(
                f"The PLY file did not contain readable points: "
                f"{self.fused_point_cloud}")

        point_count = len(self.cloud.points)
        log(
            f"Point cloud loaded | points={point_count:,}"
            f" | colors={'yes' if self.cloud.has_colors() else 'no'}"
            f" | normals={'yes' if self.cloud.has_normals() else 'no'}"
            f" | time={elapsed_since(started_at)}")

        if point_count < 3:
            raise RuntimeError(
                "At least three points are required to create a triangle mesh.")

        """
        Not Viable Obtion takes too long and some sources find the technique
        inadequate for photogrametry
        """

    def BALLPointmethod(self) -> None:
        if self.cloud is None or not self.cloud.has_points():
            raise RuntimeError(
                "No point cloud is loaded. Call LoadPoinCloud() first.")

        if not self.cloud.has_normals():
            raise RuntimeError(
                "Ball Pivoting requires point normals, but the loaded cloud "
                "does not contain them.")

        # BPA(Ball-Pivoting Algorithm) Method
        log("Measuring nearest-neighbour distances")
        distances = self.cloud.compute_nearest_neighbor_distance()
        if len(distances) == 0:
            raise RuntimeError(
                "Open3D did not compute any neighbour distances.")

        avg_dist = np.mean(distances)
        if not np.isfinite(avg_dist) or avg_dist <= 0:
            raise RuntimeError(
                f"Invalid average neighbour distance: {avg_dist}. "
                "The point cloud may contain only duplicate points.")

        radius = 3 * avg_dist
        log(
            f"Ball Pivoting started | average_spacing={avg_dist:.6g}"
            f" | radii={radius:.6g},{radius * 2:.6g}")

        stage_started_at = time.perf_counter()

        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            self.cloud, o3d.utility.DoubleVector([radius, radius * 2]))
        if not bpa_mesh.has_vertices() or not bpa_mesh.has_triangles():
            raise RuntimeError(
                "Ball Pivoting produced an empty mesh. Check the normal "
                "orientation and ball radii.")

        log_mesh(
            "Ball Pivoting complete",
            bpa_mesh,
            stage_started_at,)

        log("Mesh simplification started | target_triangles=100,000")
        stage_started_at = time.perf_counter()
        dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
        log_mesh(
            "Mesh simplification complete",
            dec_mesh,
            stage_started_at,)

        dec_mesh.remove_degenerate_triangles()
        dec_mesh.remove_duplicated_triangles()
        dec_mesh.remove_duplicated_vertices()
        dec_mesh.remove_non_manifold_edges()

        if not dec_mesh.has_vertices() or not dec_mesh.has_triangles():
            raise RuntimeError("Mesh cleanup removed all usable geometry.")

        log_mesh("Mesh cleanup complete", dec_mesh, topology=True)

        log(f"Exporting mesh | format=OBJ | path={self.bpa_mesh}")
        write_succeeded = o3d.io.write_triangle_mesh(
            str(self.bpa_mesh), dec_mesh)
        if not write_succeeded:
            raise RuntimeError(
                f"Open3D failed to write the OBJ file: {self.bpa_mesh}")
        if not self.bpa_mesh.is_file() or self.bpa_mesh.stat().st_size == 0:
            raise RuntimeError(
                f"OBJ export reported success but produced no data: {self.bpa_mesh}")

        log(
            f"Export complete | size={self.bpa_mesh.stat().st_size / (1024 * 1024):.2f} MiB")

    def PoissonMeshGeneration(self) -> None:
        # Poisson Method
        if self.cloud is None or not self.cloud.has_points():
            raise RuntimeError(
                "No point cloud is loaded. Call LoadPoinCloud() first.")
        if not self.cloud.has_normals():
            raise RuntimeError(
                "Poisson reconstruction requires point normals, but the "
                "loaded cloud does not contain them.")
        if not 0 <= self.poisson_density_quantile < 1:
            raise ValueError(
                "poisson_density_quantile must be at least 0 and less than 1.")
        if not 0 < self.mesh_target_face_ratio <= 1:
            raise ValueError(
                "mesh_target_face_ratio must be greater than 0 and at most 1.")

        started_at = time.perf_counter()
        log(
            "Poisson reconstruction started"
            " | depth=10 | scale=1.1 | samples_per_node=5"
            " | point_weight=2 | threads=all")

        stage_started_at = time.perf_counter()
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.cloud,
            depth=10,
            samples_per_node=5.0,)

        if not poisson_mesh.has_vertices() or not poisson_mesh.has_triangles():
            raise RuntimeError(
                "Poisson reconstruction produced an empty mesh.")

        log_mesh(
            "Poisson reconstruction complete",
            poisson_mesh,
            stage_started_at,
            topology=True,)

        density_values = np.asarray(densities)
        if density_values.size != len(poisson_mesh.vertices):
            raise RuntimeError(
                "Poisson returned a density count that does not match its vertices.")

        density_threshold = np.quantile(
            density_values,
            self.poisson_density_quantile,)

        low_density_mask = density_values < density_threshold
        removed_vertex_count = int(np.count_nonzero(low_density_mask))
        log(
            f"Density filter started | quantile={self.poisson_density_quantile:.1%}"
            f" | threshold={density_threshold:.4g}"
            f" | vertices_to_remove={removed_vertex_count:,}"
            f" | density_min/median/max={density_values.min():.4g}/"
            f"{np.median(density_values):.4g}/{density_values.max():.4g}")

        poisson_mesh.remove_vertices_by_mask(low_density_mask)
        if not poisson_mesh.has_vertices() or not poisson_mesh.has_triangles():
            raise RuntimeError(
                "Density filtering removed all Poisson mesh geometry.")

        log_mesh("Density filter complete", poisson_mesh, topology=True)

        bbox = self.cloud.get_axis_aligned_bounding_box()
        p_mesh_crop = poisson_mesh.crop(bbox)
        if not p_mesh_crop.has_vertices() or not p_mesh_crop.has_triangles():
            raise RuntimeError("Cropping removed all Poisson mesh geometry.")

        log_mesh("Mesh crop complete", p_mesh_crop, topology=True)

        source_triangle_count = len(p_mesh_crop.triangles)
        target_triangle_count = max(
            1,
            int(source_triangle_count * self.mesh_target_face_ratio),)

        log(
            f"Mesh simplification started | source_triangles={source_triangle_count:,}"
            f" | target_triangles={target_triangle_count:,}"
            f" | retained={self.mesh_target_face_ratio:.1%}")

        stage_started_at = time.perf_counter()
        simplified_mesh = p_mesh_crop.simplify_quadric_decimation(
            target_number_of_triangles=target_triangle_count,)

        if (not simplified_mesh.has_vertices()
                or not simplified_mesh.has_triangles()):
            raise RuntimeError(
                "Poisson mesh simplification produced an empty mesh.")
        log_mesh(
            "Mesh simplification complete",
            simplified_mesh,
            stage_started_at,
            topology=True,)

        simplified_mesh.remove_degenerate_triangles()
        simplified_mesh.remove_duplicated_triangles()
        simplified_mesh.remove_duplicated_vertices()
        simplified_mesh.remove_non_manifold_edges()
        simplified_mesh.remove_unreferenced_vertices()
        if (not simplified_mesh.has_vertices()
                or not simplified_mesh.has_triangles()):
            raise RuntimeError("Mesh cleanup removed all usable geometry.")

        log_mesh("Mesh cleanup complete", simplified_mesh, topology=True)

        repaired = self.repairMesh(o3d2pymeshlab(simplified_mesh))
        self.create_Tex(repaired)
        
        log(
            f"Conversion complete | output={self.simplified_mesh}"
            f" | size={self.textured_mesh.stat().st_size / (1024 * 1024):.2f} MiB"
            f" | total_time={elapsed_since(started_at)}")

    def repairMesh(self, mesh: pymeshlab.Mesh) -> o3d.geometry.TriangleMesh:
        started_at = time.perf_counter()
        log(
            f"PyMeshLab repair started | vertices={mesh.vertex_number():,}"
            f" | triangles={mesh.face_number():,}")

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)

        ms.apply_filter("meshing_remove_duplicate_faces")
        ms.apply_filter("meshing_remove_null_faces")

        ms.apply_filter(
            "meshing_repair_non_manifold_vertices",
            vertdispratio=0.0)

        ms.apply_filter("meshing_re_orient_faces_coherently")
        ms.apply_filter("meshing_remove_unreferenced_vertices")

        ms.apply_filter(
            "meshing_repair_non_manifold_edges",
            method="Split Vertices")
        repaired = pymeshlab2o3d(ms.current_mesh())
        topology = log_mesh(
            "PyMeshLab repair complete",
            repaired,
            started_at,
            topology=True,)

        assert topology is not None
        edge_manifold, vertex_manifold, _ = topology

        topology_errors = []
        if not edge_manifold:
            topology_errors.append("non-manifold edges")
        if not vertex_manifold:
            topology_errors.append("non-manifold vertices")
        if topology_errors:
            raise RuntimeError(
                "Mesh repair did not produce UV-compatible topology: "
                + ", ".join(topology_errors))

        return repaired

    def create_Tex(self, mesh: o3d.geometry.TriangleMesh) -> None:
        if not mesh.has_vertex_colors():
            raise RuntimeError(
                "Cannot bake an albedo texture because the mesh has no vertex colors.")
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        log(
            f"UV atlas started | size={self.texture_size}x{self.texture_size}"
            f" | gutter={self.texture_gutter:g}px"
        )
        started_at = time.perf_counter()
        stretch, charts, partitions = mesh.compute_uvatlas(
            size=self.texture_size,
            gutter=self.texture_gutter)

        log(f"UV atlas complete | charts={charts:,} | partitions={partitions:,}"
            f" | max_stretch={stretch:.4f}"
            f" | time={elapsed_since(started_at)}")

        mesh.vertex["albedo"] = mesh.vertex.colors
        mesh.material.set_default_properties()

        log(
            "Texture baking started | attribute=albedo"
            f" | size={self.texture_size}x{self.texture_size}"
        )
        started_at = time.perf_counter()
        textures = mesh.bake_vertex_attr_textures(
            self.texture_size,
            {"albedo"},
            margin=self.texture_gutter,
            update_material=False)

        albedo = textures["albedo"].clip(0, 1) * 255
        albedo = albedo.to(o3d.core.Dtype.UInt8)
        mesh.material.texture_maps["albedo"] = o3d.t.geometry.Image(albedo)
        log(f"Texture baking complete | time={elapsed_since(started_at)}")

        log(
            f"Exporting textured mesh | format=GLB | path={self.textured_mesh}")
        if not o3d.t.io.write_triangle_mesh(str(self.textured_mesh), mesh):
            raise RuntimeError(
                f"Failed to export textured GLB: {self.textured_mesh}")

        if (not self.textured_mesh.is_file()
                or self.textured_mesh.stat().st_size == 0):
            raise RuntimeError(
                f"GLB export reported success but produced no data: {self.textured_mesh}")

    def run(self) -> None:
        log("Point-cloud conversion started")
        self.LoadPoinCloud()
        self.PoissonMeshGeneration()


def o3d2pymeshlab(open3d_mesh: o3d.geometry.TriangleMesh) -> pymeshlab.Mesh:
    vertices = np.asarray(open3d_mesh.vertices)
    triangles = np.asarray(open3d_mesh.triangles)

    kwargs = {"vertex_matrix": vertices,
              "face_matrix": triangles}

    if open3d_mesh.has_vertex_colors():
        rgb = np.asarray(open3d_mesh.vertex_colors)
        alpha = np.ones((len(rgb), 1))
        kwargs["v_color_matrix"] = np.hstack((rgb, alpha))

    return pymeshlab.Mesh(**kwargs)


def pymeshlab2o3d(mesh: pymeshlab.Mesh) -> o3d.geometry.TriangleMesh:
    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertex_matrix()),
        o3d.utility.Vector3iVector(mesh.face_matrix()))

    if mesh.has_vertex_color():
        # PyMeshLab uses RGBA; Open3D uses RGB
        colors = mesh.vertex_color_matrix()[:, :3]
        result.vertex_colors = o3d.utility.Vector3dVector(colors)

    result.compute_triangle_normals()
    result.compute_vertex_normals()

    return result
