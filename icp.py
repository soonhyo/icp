import numpy as np
import open3d as o3d
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


# demo_icp_pcds = o3d.data.DemoICPPointClouds()
# source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
# target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

print("Testing IO for meshes ...")
source_mesh = o3d.io.read_triangle_mesh("/home/s-kim/tmp/flame-fitting/output/fit_scan_result.obj")
target_mesh = o3d.io.read_triangle_mesh("/home/s-kim/mesh_smooth.obj")o
# R = target_mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 4))
# target_mesh.rotate(R, center=(0, 0, 0))

target_mesh.translate(np.array([1, 1, 1]))

o3d.visualization.draw_geometries([source_mesh, target_mesh])

source_pcd = source_mesh.sample_points_uniformly(number_of_points=20000)
target_pcd = target_mesh.sample_points_uniformly(number_of_points=20000)

# source_pcd.compute_vertex_normals() #
# target_pcd.compute_vertex_normals()
source_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

o3d.visualization.draw_geometries([source_pcd, target_pcd])

translation_vec = source_pcd.get_center() - target_pcd.get_center()
target_pcd.translate(translation_vec)


o3d.visualization.draw_geometries([source_pcd, target_pcd])

threshold = 10
trans_init = np.identity(4)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source_pcd, target_pcd, threshold, trans_init)
print(evaluation)

print("Apply point-to-plane ICP")
reg_p2l = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-6,
                                                      relative_rmse = 1e-6,
                                                      max_iteration=5000))

print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
source_pcd.transform(reg_p2l.transformation)
# draw_registration_result(source_pcd, target_pcd, reg_p2l.transformation)
o3d.visualization.draw_geometries([source_pcd, target_pcd])

# criteria = o3d.ICPConvergenceCriteria(relative_fitness = 1e-6, # fitnessの変化分がこれより小さくなったら収束
#                                        relative_rmse = 1e-6, # RMSEの変化分がこれより小さくなったら収束
#                                        max_iteration = 1) # 反復1回だけにする
# est_method = py3d.TransformationEstimationPointToPoint()

# for i in range(30):
#     info = py3d.registration_icp(pcd1, pcd2,
#                                  max_correspondence_distance=th,
#                                  # init=T, # デフォルトで単位行列
#                                  estimation_method=est_method,
#                                  criteria=criteria
#                                 )
#     print("iteration {0:02d} fitness {1:.6f} RMSE {2:.6f}".format(i, info.fitness, info.inlier_rmse))

#     pcd1.transform(info.transformation)

#     py3d.draw_geometries([pcd1, pcd2], "iteration {}".format(i), 640, 480)
