import numpy as np
import open3d as o3d
import copy

print("Testing IO for npy ...")
source = np.load("data/source.npy")
target = np.load("data/target.npy")

source_v3v = o3d.utility.Vector3dVector(source)
target_v3v = o3d.utility.Vector3dVector(target)

source_pcd=o3d.geometry.PointCloud(source_v3v)
source_pcd.paint_uniform_color([1,0,0])#red
target_pcd=o3d.geometry.PointCloud(target_v3v)
target_pcd.paint_uniform_color([0,0,1])#blue

# 대응관계를 알고 있는 source 포인트와 target 포인트의 인덱스를 매핑한 배열을 생성합니다.
correspondences = o3d.utility.Vector2iVector(np.array([[idx, idx] for idx in range(len(source))]))
#correspondences = np.array([[idx, idx] for idx in range(len(source))])

reg = o3d.pipelines.registration.TransformationEstimationPointToPoint()
trans = reg.compute_transformation(source_pcd, target_pcd, correspondences)
result = copy.deepcopy(source_pcd)
result = result.transform(trans)
# ICP 알고리즘을 사용하여 정합을 수행합니다.

result.paint_uniform_color([0,1,0]) #green

print("transformamtion: ", trans)
o3d.visualization.draw_geometries([source_pcd, target_pcd, result])
