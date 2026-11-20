import mujoco
import numpy as np

from mujoco_lidar.multiray_compat import multi_ray


class StaticFieldMesh:
    """One triangle index shared by the CPU workers of the 2027 LiDAR process."""

    @classmethod
    def from_model(cls, model, geomgroup, bodyexclude):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc_2027_field_visual")
        if geom < 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            return None
        group = model.geom_group[geom]
        # The native query can omit a whole group. Keep it native if it also
        # contains a fixture, or the field itself is excluded from this sensor.
        if (model.geom_bodyid[geom] != 0 or bodyexclude == 0 or not geomgroup[group]
                or np.count_nonzero(model.geom_group == group) != 1):
            return None
        material = model.geom_matid[geom]
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
        if rgba[3] == 0:
            return None
        try:
            from embreex.mesh_construction import TriangleMesh
            from embreex.rtcore_scene import EmbreeScene
        except ImportError:
            return None

        mesh = model.mesh(model.geom_dataid[geom])
        vertices = model.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0] + mesh.vertnum[0]]
        faces = model.mesh_face[mesh.faceadr[0]:mesh.faceadr[0] + mesh.facenum[0]]
        scene = EmbreeScene()
        geometry = TriangleMesh(scene, vertices, faces)
        # Embree commits on first use. Finish construction before workers share it.
        scene.run(np.zeros((1, 3), dtype="f"), np.array([[1, 0, 0]], dtype="f"))
        return cls(geom, group, model.geom_rbound[geom], vertices, faces, scene, geometry)

    def __init__(self, geom, group, radius, vertices, faces, scene, geometry):
        self.geom = geom
        self.group = group
        self.radius = radius
        self.vertices = vertices
        self.faces = faces
        self.scene = scene
        self.geometry = geometry

    def merge_distances(self, data, point, directions, cutoff, geom_ids, distances):
        offset = point - data.geom_xpos[self.geom]
        if np.linalg.norm(offset) > cutoff + self.radius:
            return
        rotation = data.geom_xmat[self.geom].reshape(3, 3)
        origin = offset @ rotation
        local_dirs = directions @ rotation
        origins = np.tile(origin.astype(np.float32), (len(directions), 1))
        triangles = self.scene.run(origins, np.ascontiguousarray(local_dirs, dtype=np.float32))
        hits = np.flatnonzero(triangles >= 0)
        # Use the index for face selection and the original MuJoCo triangle for range.
        a, b, c = self.vertices[self.faces[triangles[hits]]].astype(np.float64).transpose(1, 0, 2)
        normals = np.cross(b - a, c - a)
        ranges = np.einsum("ij,ij->i", a - origin, normals) / np.einsum(
            "ij,ij->i", local_dirs[hits], normals
        )
        # A fixture or another native geom can be closer than the field mesh.
        closer = (ranges >= 0) & ((geom_ids[hits] == -1) | (ranges < distances[hits]))
        distances[hits[closer]] = ranges[closer]
        geom_ids[hits[closer]] = self.geom


class MjLidarCPU:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        cutoff_dist: float = 100.0,
        geomgroup: np.ndarray | None = None,
        bodyexclude: int = -1,
        field_mesh: StaticFieldMesh | None = None,
    ) -> None:

        self.mj_model = mj_model
        self.cutoff_dist = cutoff_dist
        self.geomgroup = geomgroup
        self.bodyexclude = bodyexclude
        self.field_mesh = field_mesh
        if field_mesh is not None:
            self.geomgroup = self.geomgroup.copy()
            self.geomgroup[field_mesh.group] = 0

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
        if self.field_mesh is not None:
            self.field_mesh.merge_distances(
                self.mj_data, pnt.ravel(), world_vecs_flat.reshape(-1, 3),
                self.cutoff_dist, _geomid, self._dist,
            )
        self._dist[_geomid == -1] = 0

        # Update the pcl frame with local frame data
        self._hit_points = local_vecs * self._dist[:, np.newaxis]

    def get_hit_points(self) -> np.ndarray | None:
        return self._hit_points

    def get_distances(self) -> np.ndarray | None:
        return self._dist
