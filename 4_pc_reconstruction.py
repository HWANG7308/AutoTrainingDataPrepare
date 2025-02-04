import os
import sys
from tqdm import tqdm
from utils.PointCloudProcessor import PointCloudProcessor


def generate_point_clouds(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for obj in tqdm(os.listdir(data_dir)):
        rgb_dir = os.path.join(data_dir, obj, "color")
        pc_save_dir = os.path.join(output_dir, obj)
        os.makedirs(pc_save_dir, exist_ok=True)

        pc_processor = PointCloudProcessor(rgb_dir=rgb_dir)
        pcd = pc_processor.reconstruct_point_cloud()
        pcd = pc_processor.post_process_point_cloud(
            pcd,
            voxel_size=5e-4,
        )
        pc_processor.save_point_cloud(
            pcd, save_dir=pc_save_dir, filename="pc", save_format="ply"
        )

        mesh = pc_processor.reconstruct_mesh_from_point_cloud(pcd)
        pc_processor.save_mesh(
            mesh, save_dir=pc_save_dir, filename="mesh", save_format="ply"
        )


if __name__ == "__main__":
    data_dir = "./result/acquired_data"
    output_dir = "./result/reconstructed_3d_models"

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        sys.exit(1)

    generate_point_clouds(data_dir, output_dir)
