import numpy as np
import transforms3d as t3d
import transforms3d as t3d
from probreg import filterreg
from probreg import features
from probreg import callbacks
import utils
import open3d as o3d
import time

source_mesh = o3d.io.read_triangle_mesh("/home/s-kim/tmp/flame-fitting/output/fit_scan_result.obj")
target_mesh = o3d.io.read_triangle_mesh("/home/s-kim/mesh_smooth.obj")
target_mesh.translate(np.array([1, 1, 1]))

source_pcd = source_mesh.sample_points_uniformly(number_of_points=1000)
target_pcd = target_mesh.sample_points_uniformly(number_of_points=1000)

source_pcd.remove_non_finite_points()
target_pcd.remove_non_finite_points()

# source_pcd = source_pcd.voxel_down_sample(voxel_size=0.001)
# target_pcd = target_pcd.voxel_down_sample(voxel_size=0.001)
R = target_pcd.get_rotation_matrix_from_xyz((np.pi/180, 0, 0))

cbs = [callbacks.Open3dVisualizerCallback(source_pcd, target_pcd)]

while True:
    target_pcd.rotate(R, center=(0,0,0))

    translation_vec =  target_pcd.get_center() -source_pcd.get_center()
    source_pcd.translate(translation_vec)
    objective_type = 'pt2pt'
    tf_param, _, _ = filterreg.registration_filterreg(source_pcd, target_pcd,
                                                      objective_type=objective_type,
                                                      sigma2=1000, feature_fn=features.FPFH(),
                                                      callbacks=cbs)

    print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
          tf_param.scale, tf_param.t)
