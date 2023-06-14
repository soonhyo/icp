import copy
import numpy as np
use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x
import open3d as o3d
from probreg import cpd
from probreg import callbacks
import transforms3d as trans
import utils
# load source and target point cloud
# source = o3d.io.read_point_cloud('bunny_ascii.pcd')
source_mesh = o3d.io.read_triangle_mesh("/home/s-kim/tmp/flame-fitting/output/fit_scan_result.obj")
target_mesh = o3d.io.read_triangle_mesh("/home/s-kim/mesh_smooth.obj")
target_mesh.translate(np.array([1, 1, 1]))

source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)
target_pcd = target_mesh.sample_points_uniformly(number_of_points=5000)

source_pcd.remove_non_finite_points()
target_pcd.remove_non_finite_points()

source_pcd = source_pcd.voxel_down_sample(voxel_size=0.001)
target_pcd = target_pcd.voxel_down_sample(voxel_size=0.001)

translation_vec = source_pcd.get_center() - target_pcd.get_center()
target_pcd.translate(translation_vec)

source_pt = cp.asarray(source_pcd.points, dtype=cp.float32)
target_pt = cp.asarray(target_pcd.points, dtype=cp.float32)

# transform target point cloud
# th = np.deg2rad(30.0)
# target_pcd.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
#                            [np.sin(th), np.cos(th), 0.0, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]]))

# compute cpd registration
rcpd = cpd.RigidCPD(source_pt, use_cuda=use_cuda)
# acpd = cpd.AffineCPD(source_pt, use_cuda=use_cuda)
# cbs = [callbacks.Open3dVisualizerCallback(source_pt, target_pt)]
# tf_param, _ = gmmtree.registration_gmmtree(source_pt, target_pt, callbacks=cbs)

tf_param, _, _ = rcpd.registration(target_pt)

result = tf_param.transform(source_pt)

result_pcd = o3d.geometry.PointCloud()
result_pcd.points = o3d.utility.Vector3dVector(to_cpu(result))

# draw result
source_pcd.paint_uniform_color([1, 0, 0])
target_pcd.paint_uniform_color([0, 1, 0])
result_pcd.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([source_pcd, target_pcd, result_pcd])
