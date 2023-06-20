import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from haircutrobot.msg import *
import pandas as pd
import util
import sensor_msgs.point_cloud2 as pc2

def lmk2dict(data):
    result ={}
    for d in data:
        result[d.index] = [d.x,d.y,d.z]
    return result

def publish_pointcloud(output_data, header):
    # convert pcl data format
    pc_p = np.asarray(output_data.points)
    # pc_c = np.asarray(output_data.colors)
    # tmp_c = np.c_[np.zeros(pc_c.shape[1])]
    # tmp_c = np.floor(pc_c[:,0] * 255) * 2**16 + np.floor(pc_c[:,1] * 255) * 2**8 + np.floor(pc_c[:,2] * 255) # 16bit shift, 8bit shift, 0bit shift

    # pc_pc = np.c_[pc_p, tmp_c]
    # print(pc_pc)
    # publish point cloud
    output = pc2.create_cloud(header, FIELDS , pc_p)
    pub.publish(output)

def callback(msg):
    # ROS 메시지인 PointCloud2를 Open3D 포인트 클라우드로 변환합니다.
    landmarks_dict = lmk2dict(msg.landmarks)

    landmarks_values = np.array(list(landmarks_dict.values()))
    # print(df["fan"].values)
    landmarks_51 = landmarks_values[df["fan"].values]

    landmarks_inf = np.copy(landmarks_51)
    landmarks_inf[np.all(landmarks_inf<0.0001, axis=1)] =np.inf

    print(landmarks_inf.shape)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(landmarks_inf)

    # target 포인트 클라우드를 로드합니다.
    target = np.load("data/target.npy")
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)

    # 대응 관계를 알고 있는 source 포인트와 target 포인트의 인덱스를 매핑한 배열을 생성합니다.
    correspondences = o3d.utility.Vector2iVector(np.array([[idx, idx] for idx in range(len(cloud.points))]))

    # ICP 알고리즘을 사용하여 정합을 수행합니다.
    reg = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans = reg.compute_transformation(target_pcd, cloud,correspondences)
    aligned_cloud = target_pcd.transform(trans)
    aligned_cloud.paint_uniform_color([1,0,0]) #red
    # 정합된 포인트 클라우드를 ROS 메시지로 변환합니다.
    header = Header()
    header.stamp = rospy.Time.now()
    #header.frame_id = "camera_color_optical_frame"
    header.frame_id = "head_link"

    publish_pointcloud(aligned_cloud,header)

rospy.init_node('icp_node')
df = pd.read_excel('mp2fan.xlsx')

FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    # PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
]

# /landmarks 토픽을 구독하고 메시지가 수신될 때마다 callback 함수를 호출합니다.
rospy.Subscriber('/face_mesh_node/landmarks', LandmarkArray, callback)

# /aligned_landmarks 토픽에 정합된 포인트 클라우드를 발행합니다.
pub = rospy.Publisher('/aligned_landmarks', PointCloud2, queue_size=10)

rospy.spin()
