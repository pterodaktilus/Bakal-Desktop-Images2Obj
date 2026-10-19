import re
import shutil
import subprocess
from pathlib import Path

from config import ColmapsConfig


class ColmapsReconstruction(ColmapsConfig):
    """COLMAP reconstruction, meshing, and export stages."""

    def __init__(self) -> None:
        super().__init__()
        self.selected_image_names: list[str] = []

    @staticmethod
    def _is_nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def _run_colmap(self, command: str, *arguments: object, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
        colmap_arguments = ["colmap", command]
        colmap_arguments.extend(str(argument) for argument in arguments)
        print(f"Running COLMAP: {command}", flush=True)

        try:
            return subprocess.run(
                colmap_arguments,
                check=True,
                capture_output=capture_output,
                text=True,
            )
        except OSError as error:
            raise RuntimeError(
                f"Could not start COLMAP command '{command}': {error}"
            ) from error
        except subprocess.CalledProcessError as error:
            details = "\n".join(
                output.strip()
                for output in (error.stdout, error.stderr)
                if output and output.strip()
            )
            message = (
                f"COLMAP command '{command}' failed with exit code "
                f"{error.returncode}."
            )
            if details:
                message = f"{message}\n{details}"
            raise RuntimeError(message) from error

    def _write_selected_image_list(self) -> None:
        if not self.selected_image_names:
            raise RuntimeError("No images were selected for reconstruction.")
        self.image_list.write_text(
            "\n".join(self.selected_image_names) + "\n",
            encoding="utf-8",
        )

    def _model_statistics(self, model_path: Path) -> tuple[int, int]:
        completed = self._run_colmap(
            "model_analyzer",
            "--path",
            model_path,
            capture_output=True,
        )
        output = "\n".join((completed.stdout or "", completed.stderr or ""))
        registered_match = re.search(r"Registered images:\s*(\d+)", output)
        points_match = re.search(r"Points:\s*(\d+)", output)
        if registered_match is None:
            raise RuntimeError(
                f"Could not read the registered-image count for model {model_path}.\n"
                f"COLMAP output:\n{output.strip()}"
            )
        return (
            int(registered_match.group(1)),
            int(points_match.group(1)) if points_match else 0,
        )

    def _select_best_sparse_model(self) -> tuple[Path, int, int]:
        model_paths = sorted(
            path
            for path in self.sparse.iterdir()
            if path.is_dir()
            and any(
                (path / f"cameras.{extension}").is_file()
                for extension in ("bin", "txt")
            )
        )
        if not model_paths:
            raise RuntimeError("COLMAP failed to reconstruct any model.")

        analyzed_models = [
            (path, *self._model_statistics(path)) for path in model_paths
        ]
        return max(analyzed_models, key=lambda model: (model[1], model[2]))

    def reconstruct_sparse_model(self) -> None:
        self._write_selected_image_list()

        print("Extracting features...")
        self._run_colmap(
            "feature_extractor",
            "--database_path",
            self.database,
            "--image_path",
            self.images,
            "--image_list_path",
            self.image_list,
            "--ImageReader.camera_model",
            self.camera_model,
            "--ImageReader.single_camera",
            self.single_camera,
            "--FeatureExtraction.use_gpu",
            self.feature_extraction_use_gpu,
            "--FeatureExtraction.gpu_index",
            self.feature_extraction_gpu_index,
            "--FeatureExtraction.num_threads",
            self.feature_extraction_num_threads,
            "--FeatureExtraction.max_image_size",
            self.feature_extraction_max_image_size,
            "--SiftExtraction.max_num_features",
            self.sift_max_num_features,
            "--SiftExtraction.estimate_affine_shape",
            self.sift_estimate_affine_shape,
            "--SiftExtraction.domain_size_pooling",
            self.sift_domain_size_pooling,
        )

        print("Matching image pairs...")
        self._run_colmap(
            self.feature_matcher,
            "--database_path",
            self.database,
            "--FeatureMatching.use_gpu",
            self.feature_matching_use_gpu,
            "--FeatureMatching.gpu_index",
            self.feature_matching_gpu_index,
            "--FeatureMatching.num_threads",
            self.feature_matching_num_threads,
            "--FeatureMatching.guided_matching",
            self.feature_matching_guided_matching,
        )

        print("Creating sparse reconstruction...")
        self._run_colmap(
            "mapper",
            "--database_path",
            self.database,
            "--image_path",
            self.images,
            "--output_path",
            self.sparse,
            "--Mapper.image_list_path",
            self.image_list,
            "--Mapper.multiple_models",
            self.mapper_multiple_models,
            "--Mapper.min_model_size",
            self.mapper_min_model_size,
            "--Mapper.init_min_tri_angle",
            self.mapper_init_min_tri_angle,
            "--Mapper.ba_local_max_num_iterations",
            self.mapper_ba_local_max_num_iterations,
            "--Mapper.ba_local_max_refinements",
            self.mapper_ba_local_max_refinements,
            "--Mapper.ba_global_max_num_iterations",
            self.mapper_ba_global_max_num_iterations,
        )

        best_model, registered_images, sparse_points = (
            self._select_best_sparse_model()
        )
        shutil.copytree(best_model, self.model)
        print(
            f"Using model {best_model.name}: {registered_images} registered images, "
            f"{sparse_points} sparse points"
        )
        print("Sparse model:", self.model)

    def prepare_dense_workspace(self) -> set[str]:
        print("Undistorting images...")
        self._run_colmap(
            "image_undistorter",
            "--image_path",
            self.images,
            "--input_path",
            self.model,
            "--output_path",
            self.dense,
            "--output_type",
            "COLMAP",
            "--max_image_size",
            self.max_image_size,
        )

        undistorted_images = self.dense / "images"
        registered_image_names = {
            path.relative_to(undistorted_images).as_posix()
            for path in undistorted_images.rglob("*")
            if path.is_file()
        }
        if not registered_image_names:
            raise RuntimeError(
                "The sparse reconstruction has no registered images.")
        return registered_image_names

    def create_depth_maps(self, registered_image_names: set[str]) -> None:
        print("Running PatchMatch stereo...")
        self._run_colmap(
            "patch_match_stereo",
            "--workspace_path",
            self.dense,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.max_image_size",
            self.max_image_size,
            "--PatchMatchStereo.gpu_index",
            self.patch_match_gpu_index,
            "--PatchMatchStereo.geom_consistency",
            self.patch_match_geom_consistency,
            "--PatchMatchStereo.window_step",
            self.patch_match_window_step,
            "--PatchMatchStereo.num_samples",
            self.patch_match_num_samples,
            "--PatchMatchStereo.num_iterations",
            self.patch_match_num_iterations,
        )

        output_maps_are_complete = all(
            self._is_nonempty_file(
                self.depth_maps
                / f"{image_name}.{self.stereo_fusion_input_type}.bin"
            )
            and self._is_nonempty_file(
                self.normal_maps
                / f"{image_name}.{self.stereo_fusion_input_type}.bin"
            )
            for image_name in registered_image_names
        )
        if not output_maps_are_complete:
            raise RuntimeError(
                "PatchMatch finished without all expected output maps.")

    def fuse_point_cloud(self) -> Path:
        print("Fusing point cloud...")
        self._run_colmap(
            "stereo_fusion",
            "--workspace_path",
            self.dense,
            "--workspace_format",
            "COLMAP",
            "--input_type",
            self.stereo_fusion_input_type,
            "--output_type",
            "PLY",
            "--output_path",
            self.fused_point_cloud,
        )
        if not self._is_nonempty_file(self.fused_point_cloud):
            raise RuntimeError("Stereo fusion did not produce a point cloud.")
        print("Dense cloud:", self.fused_point_cloud)
        return self.fused_point_cloud

    def create_mesh(self) -> None:
        print("Creating Poisson mesh...")
        self._run_colmap(
            "poisson_mesher",
            "--input_path",
            self.fused_point_cloud,
            "--output_path",
            self.poisson_mesh,
        )
        if not self._is_nonempty_file(self.poisson_mesh):
            raise RuntimeError(
                "Poisson meshing did not produce an output mesh.")

        print("Simplifying mesh for Unity...")
        self._run_colmap(
            "mesh_simplifier",
            "--input_path",
            self.poisson_mesh,
            "--output_path",
            self.simplified_mesh,
            "--MeshSimplification.target_face_ratio",
            self.mesh_target_face_ratio,
        )
        if not self._is_nonempty_file(self.simplified_mesh):
            raise RuntimeError(
                "Mesh simplification did not produce an output mesh.")

        print("Dense mesh:", self.poisson_mesh)
        print("Plain simplified mesh:", self.simplified_mesh)

    def run(self) -> None:
        """Execute every reconstruction stage after workspace preparation."""
        self.reconstruct_sparse_model()
        registered_image_names = self.prepare_dense_workspace()
        self.create_depth_maps(registered_image_names)
        self.fuse_point_cloud()
        # self.create_mesh()

    def run_auto(self) -> None:
        self._run_colmap(
            "automatic_reconstructor",
            "--workspace_path",
            self.output,
            "--quality",
            "EXTREME",
            "--single_camera",
            1,
            "--image_list_path",
            self.images,
            "--output_type",
            "PLY",
            "--data_type", "INTERNET",
            "--camera_model", "OPENCV"
        )
