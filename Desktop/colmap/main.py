import shutil
from pathlib import Path

from ColmapsReconstruction import ColmapsReconstruction
from ColmapMeshPipeline import ColmapMeshPipeline


class Reconstruct(ColmapsReconstruction):
    """Prepare the workspace and coordinate the reconstruction pipeline."""
    IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff",".dng"}

    def prepare_workspace(self) -> None:
        if not self.images.is_dir():
            raise RuntimeError(
                f"Image directory does not exist: {self.images}"
            )
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")

        if not 0 < self.mesh_target_face_ratio <= 1:
            raise ValueError(
                "mesh_target_face_ratio must be greater than 0 and at most 1"
            )

        all_image_names = sorted(
            path.name
            for path in self.images.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if not all_image_names:
            raise RuntimeError(f"No images found in {self.images}")

        self.selected_image_names = all_image_names[::self.frame_stride]
        if self.selected_image_names[-1] != all_image_names[-1]:
            self.selected_image_names.append(all_image_names[-1])

        print(f"Using {len(self.selected_image_names)} of {len(all_image_names)} images "
              f"(frame stride: {self.frame_stride})")

        self.output.mkdir(parents=True, exist_ok=True)
        print("Starting a clean reconstruction...")
        if self.database.exists():
            self.database.unlink()
        for path in (self.sparse, self.dense, self.model):
            if path.exists():
                shutil.rmtree(path)

        self.sparse.mkdir(exist_ok=True)
        self.dense.mkdir(exist_ok=True)


def run_pipeline() -> Path:
    reconstruction = Reconstruct()
    #reconstruction.prepare_workspace()
    #reconstruction.run()
    mesh_pipeline = ColmapMeshPipeline()
    mesh_pipeline.run(mesher="open3d")
    return reconstruction.output


if __name__ == "__main__":
    run_pipeline()

