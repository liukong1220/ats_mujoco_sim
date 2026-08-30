import mujoco
import numpy as np

from mujoco_lidar.multiray_compat import multi_ray


class MjLidarCPU:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        cutoff_dist: float = 100.0,
        geomgroup: np.ndarray | None = None,
        bodyexclude: int = -1,
    ) -> None:

        self.mj_model = mj_model
        self.cutoff_dist = cutoff_dist
        self.geomgroup = geomgroup
        self.bodyexclude = bodyexclude

        self._dist: np.ndarray | None = None
        self._hit_points: np.ndarray | None = None

    def update(self, mj_data: mujoco.MjData) -> None:
        self.mj_data = mj_data

    def trace_rays(self, pose_4x4: np.ndarray, ray_theta: np.ndarray, ray_phi: np.ndarray) -> None:

        if ray_phi.shape[0] != ray_theta.shape[0]:
            raise ValueError("ray_phi and ray_theta must have the same shape")

        _nray = ray_phi.shape[0]

        # Initialize
        self._dist = np.full(_nray, self.cutoff_dist, dtype=np.float64)
        _geomid = np.full(_nray, 0, dtype=np.int32)

        # Uniformly generate vec from site's pose and lidar settings
        # Note that all the vec are in the local frame.
        site_pos, site_mat = pose_4x4[:3, 3], pose_4x4[:3, :3]
        # mujoco>=3.2 的 mj_multiRay 绑定要求 pnt/vec 是 float64；site 位姿由
        # MuJoCo 以 float32 传入，必须显式提升，否则抛 TypeError 并使整个
        # 雷达子进程退出（表现为闭环里 /local_pointcloud 永不发布）。
        pnt = np.array([site_pos], dtype=np.float64).T
        x = np.cos(ray_phi) * np.cos(ray_theta)
        y = np.cos(ray_phi) * np.sin(ray_theta)
        z = np.sin(ray_phi)
        local_vecs = np.stack((x, y, z), axis=-1)
        world_vecs = local_vecs @ site_mat.T
        world_vecs /= np.linalg.norm(world_vecs, axis=1, keepdims=True)
        world_vecs_flat = np.ascontiguousarray(world_vecs.flatten(), dtype=np.float64)

        # Get the ray casting results.
        # The optional ``normal`` slot exists in MuJoCo 3.10 but not in 3.4, so
        # the slot count is probed at runtime instead of pinned to one release.
        # Getting it wrong aborts the LiDAR child before it can publish
        # /registered_scan, which surfaces as missing localization rather than
        # as an API mismatch.
        multi_ray(
            self.mj_model,
            self.mj_data,
            pnt,
            world_vecs_flat,
            self.geomgroup,
            1,
            self.bodyexclude,
            _geomid,
            self._dist,
            _nray,
            self.cutoff_dist,
        )
        # Calculate the point's position in local frame from vec + dist
        self._dist[_geomid == -1] = 0

        # Update the pcl frame with local frame data
        self._hit_points = local_vecs * self._dist[:, np.newaxis]

    def get_hit_points(self) -> np.ndarray | None:
        return self._hit_points

    def get_distances(self) -> np.ndarray | None:
        return self._dist
