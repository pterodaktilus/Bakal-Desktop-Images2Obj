import shutil
from pathlib import Path
import sys

from something.ColmapsReconstruction import ColmapsReconstruction
from something.ColmapMeshPipeline import ColmapMeshPipeline


class Reconstruct(ColmapsReconstruction):
    """Prepare the workspace and coordinate the reconstruction pipeline."""
    IMAGE_EXTENSIONS = {".bmp", ".jpeg",
                        ".jpg", ".png", ".tif", ".tiff", ".dng"}

    def __init__(self, dataset_dir, out_dir):
        super().__init__(dataset_dir, out_dir)

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


def run_pipeline(dataset_dir: Path, output_dir: Path) -> Path:
    reconstruction = Reconstruct(dataset_dir, output_dir)
    reconstruction.prepare_workspace()
    reconstruction.run()
    mesh_pipeline = ColmapMeshPipeline(dataset_dir, output_dir)

    return mesh_pipeline.run(mesher="open3d")


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) < 3:
        exit(-1)
    dataset_path = Path(arguments[1]).resolve()
    out_path = Path(arguments[2]).resolve()

    run_pipeline(dataset_path, out_path)
