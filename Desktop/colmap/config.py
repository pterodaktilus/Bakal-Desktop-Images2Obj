from pathlib import Path


class ColmapsConfig:
    def __init__(self) -> None:
        # --------------------------------------------------
        # Paths
        # --------------------------------------------------
        self.images = Path.cwd() / "imgs"

        self.output = Path.cwd() / "done"

        self.database = self.output / "database.db"
        self.image_list = self.output / "selected-images.txt"
        self.sparse = self.output / "sparse"
        self.dense = self.output / "dense"
        self.model = self.output / "model"

        self.depth_maps = self.dense / "stereo" / "depth_maps"
        self.normal_maps = self.dense / "stereo" / "normal_maps"

        self.fused_point_cloud = self.dense / "fused.ply"
        self.bpa_mesh = self.dense / "bpa_mesh.obj"
        self.o3d_poisson_mesh = self.dense / "p_mesh_c.obj"
        self.textured_mesh = self.dense / "textured_mesh.glb"
        self.poisson_mesh = self.dense / "meshed-poisson.ply"
        self.delaunay_mesh = self.dense / "meshed-delaunay.ply"
        self.advancing_front_mesh = self.dense / "meshed-advancing-front.ply"
        self.open3d_mesh = self.dense / "meshed-open3d.ply"
        self.simplified_mesh = self.dense / "meshed-poisson-simplified.ply"

        # --------------------------------------------------
        # Reconstruction options
        # --------------------------------------------------

        # Use every Nth video frame throughout the reconstruction. The original images
        # remain untouched. Always include the last frame to preserve trajectory coverage.
        self.frame_stride = 1

        # Treat all input images as coming from one camera. Change the camera model when
        # the input lens requires a different COLMAP model.
        self.camera_model = "OPENCV"
        self.single_camera = 1

        # Limit the image size used by the dense reconstruction. A larger value preserves
        # more detail but consumes more memory and processing time.
        self.max_image_size = 1000

        # Select which PatchMatch depth maps stereo fusion consumes.
        self.stereo_fusion_input_type = "photometric"

        # COLMAP Poisson surface reconstruction and texture-atlas settings.
        self.poisson_trim = 10.0
        self.texture_size = 1024
        self.texture_gutter = 2.0

        # Keep this fraction of mesh faces for the Unity-oriented output. Lower values
        # produce a smaller mesh at the cost of geometric detail.
        self.mesh_target_face_ratio = 0.25

        # Remove this lowest-density fraction of Poisson vertices before cropping and
        # simplification. These vertices have the weakest point-cloud support.
        self.poisson_density_quantile = 0.01

        # Low-quality feature extraction options. The installed COLMAP build uses
        # the FeatureExtraction.* option names for device and image-size settings.
        self.feature_extraction_use_gpu = 1
        self.feature_extraction_gpu_index = "0"
        self.feature_extraction_num_threads = -1
        self.feature_extraction_max_image_size = 1000
        self.sift_max_num_features = 2048
        self.sift_estimate_affine_shape = 0
        self.sift_domain_size_pooling = 0

        # Exhaustive matching is practical after frame_stride reduces this video to
        # several hundred images and gives every selected image a chance to connect.
        self.feature_matcher = "exhaustive_matcher"
        self.feature_matching_use_gpu = 1
        self.feature_matching_gpu_index = "0"
        self.feature_matching_num_threads = -1
        self.feature_matching_guided_matching = 0

        # COLMAP's video-oriented initialization and low-quality bundle adjustment.
        # Keep multiple models during diagnosis so the best reconstruction can be chosen.
        self.mapper_multiple_models = 1
        self.mapper_min_model_size = 10
        self.mapper_init_min_tri_angle = 8
        self.mapper_ba_local_max_num_iterations = 12
        self.mapper_ba_local_max_refinements = 2
        self.mapper_ba_global_max_num_iterations = 25

        # Low-quality dense stereo settings trade detail for lower runtime and GPU use.
        self.patch_match_gpu_index = "0"
        self.patch_match_geom_consistency = 0
        self.patch_match_window_step = 2
        self.patch_match_num_samples = 7
        self.patch_match_num_iterations = 3
