import numpy as np
from codebase.z_utils.Rotation import euler2mat, mat2quat, quat2mat, quat2euler
from copy import copy


# def rot3_from_O_to_AB(O, A, B):
#     # 1. 单位化
#     v = O / np.linalg.norm(O)
#     w = (B - A) / np.linalg.norm(B - A)
#     d = np.dot(v, w)

#     # 2. 平行或反向
#     if np.isclose(d, 1.0):
#         return np.eye(3)
#     if np.isclose(d, -1.0):
#         # 180° 旋转，任选一个与 v 不共线的轴
#         axis = np.cross(v, [1, 0, 0])
#         if np.linalg.norm(axis) < 1e-6:
#             axis = np.cross(v, [0, 1, 0])
#         axis /= np.linalg.norm(axis)
#         K = np.array([[0, -axis[2], axis[1]],
#                       [axis[2], 0, -axis[0]],
#                       [-axis[1], axis[0], 0]])
#         return np.eye(3) + 2 * (K @ K)

#     # 3. 常规 Rodrigues
#     k = np.cross(v, w)
#     k /= np.linalg.norm(k)
#     theta = np.arccos(d)
#     K = np.array([[0, -k[2], k[1]],
#                   [k[2], 0, -k[0]],
#                   [-k[1], k[0], 0]])
#     return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# def rotate_camera_body_frame(cameraPose, Euler_rot_change):
#     """
#     Rotate the camera in the body frame.
#     :param cameraPose: [x, y, z, qx, qy, qz, qw]
#     :param Euler_rot_change: [roll, pitch, yaw]
#     :return: new camera pose
#     """
#     # Convert quaternion to rotation matrix
#     R_camera_old = quat2mat(cameraPose[3:])

#     # Convert Euler angles to rotation matrix
#     Matrix_rot_change = euler2mat(np.radians(Euler_rot_change))

#     # Calculate new rotation matrix
#     R_camera_new = R_camera_old @ Matrix_rot_change

#     # Convert rotation matrix back to quaternion
#     new_quat = mat2quat(R_camera_new)

#     # Return new camera pose
#     return np.concatenate((cameraPose[:3], new_quat))


# def rotate_camera_world_frame(cameraPose, Euler_rot_change):
#     """
#     Rotate the camera in the world frame.
#     :param cameraPose: [x, y, z, qx, qy, qz, qw]
#     :param Euler_rot_change: [roll, pitch, yaw]
#     :return: new camera pose
#     """
#     # Convert quaternion to rotation matrix
#     R_camera_old = quat2mat(cameraPose[3:])

#     # Convert Euler angles to rotation matrix
#     Matrix_rot_change = euler2mat(np.radians(Euler_rot_change))

#     # Calculate new rotation matrix
#     R_camera_new = Matrix_rot_change @ R_camera_old

#     # Convert rotation matrix back to quaternion
#     new_quat = mat2quat(R_camera_new)

#     # Return new camera pose
#     return np.concatenate((cameraPose[:3], new_quat))


# path_xy = np.load("/media/jian/ssd4t/DP/first/MV7J6NIKTKJZ2AABAAAAADA8_usd_path.npy")
# path_xy = path_xy[::11]
# z = np.zeros((path_xy.shape[0], 1))
# path_xyz = np.hstack((path_xy, z))
# O = np.array([1, 0, 0])  # 观察者的方向，假设是 z 轴正方向


# R_A = rot3_from_O_to_AB(O, path_xyz[0], path_xyz[1])
# T_A = np.eye(4)
# T_A[:3, :3] = R_A

# rotation_change_candiate = np.array([
#     [0, 30, 0],  # 低头
#     [0, 0, 0],  # still
#     [0, -30, 0],  # 抬头
# ])

# # pdb.set_trace()
# for euler_to_apply in rotation_change_candiate:
#     cameraPose_y = rotate_camera_body_frame(np.concatenate((position, rotation)), euler_to_apply)
#     # cameraPose_new = HT2eePose(T_camera_new)
#     for i in range(0, 360, 30):
#         frame_count = 0
#         image_saved = False  # 添加一个标志，确保图像只保存一次

#         right_rot = [0, 0, -i]  # [0,0,-30] [0,0,-60], ...

#         cameraPose_new = rotate_camera_world_frame(cameraPose_y, right_rot)
#         # pdb.set_trace()
#         camera.set_world_pose(position=cameraPose_new[:3], orientation=[cameraPose_new[6], cameraPose_new[3], cameraPose_new[4], cameraPose_new[5]])  # wxyz
# print()
# for cur_frame, (A, B) in enumerate(zip(path_xy[:-1], path_xy[1:])):
#     A = copy(np.array([A[0], A[1], 1]))
#     B = copy(np.array([B[0], B[1], 1]))
#     A_rot_matrix = rot3_from_O_to_AB(O, A, B)
#     orientation = Quaternion(matrix=A_rot_matrix)
#     cameraPose = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
#     position = A  # cameraPose[:3]
#     rotation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])  # cameraPose[3:]
#     euler = quat2euler(rotation)
