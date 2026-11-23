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
        # Keep native only if the field itself is non-static, excluded, or invisible.
        # Other geoms in the same group (e.g. an analytic fixture box) are handled
        # cleanly by the analytic fast path or native raycast fallback.
        if model.geom_bodyid[geom] != 0 or bodyexclude == 0 or not geomgroup[group]:
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
        raw_geomgroup = geomgroup.copy() if geomgroup is not None else None
        self.geomgroup = raw_geomgroup
        self.bodyexclude = bodyexclude
        self.field_mesh = field_mesh

        self._dist: np.ndarray | None = None
        self._hit_points: np.ndarray | None = None

        # Pre-filter active non-field geoms once using the original geomgroup so
        # co-grouped analytic fixtures (e.g. s3_test_obstacle in group 2) stay visible.
        self._analytic_geoms: list[tuple[int, int]] | None = None
        self._init_analytic_fast_path(raw_geomgroup)

        if self.field_mesh is not None:
            if self._analytic_geoms is None:
                # Native mj_multiRay fallback cannot mask a single geom without masking
                # its entire group, so disable field_mesh if another geom shares the group.
                if np.count_nonzero(mj_model.geom_group == self.field_mesh.group) != 1:
                    self.field_mesh = None
                elif self.geomgroup is not None:
                    self.geomgroup = self.geomgroup.copy()
                    self.geomgroup[self.field_mesh.group] = 0

    def _init_analytic_fast_path(self, raw_geomgroup: np.ndarray | None) -> None:
        if self.field_mesh is None or raw_geomgroup is None:
            return
        model = self.mj_model
        supported = {
            int(mujoco.mjtGeom.mjGEOM_PLANE),
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        }
        active: list[tuple[int, int]] = []
        for geom_id in range(model.ngeom):
            if geom_id == self.field_mesh.geom:
                continue
            if not raw_geomgroup[model.geom_group[geom_id]]:
                continue
            if self.bodyexclude >= 0 and model.geom_bodyid[geom_id] == self.bodyexclude:
                continue
            mat_id = model.geom_matid[geom_id]
            rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[geom_id]
            if rgba[3] == 0:
                continue
            gtype = int(model.geom_type[geom_id])
            if gtype not in supported:
                return
            active.append((geom_id, gtype))
        self._analytic_geoms = active

    def _trace_analytic_geoms(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        geom_ids: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        assert self._analytic_geoms is not None
        model = self.mj_model
        data = self.mj_data
        cutoff = self.cutoff_dist
        plane_t = int(mujoco.mjtGeom.mjGEOM_PLANE)
        sphere_t = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        box_t = int(mujoco.mjtGeom.mjGEOM_BOX)
        cyl_t = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        eps = 1.0e-12

        for geom_id, gtype in self._analytic_geoms:
            pos = data.geom_xpos[geom_id]
            rot = data.geom_xmat[geom_id].reshape(3, 3)
            size = model.geom_size[geom_id]
            rel = origin - pos
            # Bounding sphere rejection for finite primitives
            if gtype != plane_t:
                rbound = float(model.geom_rbound[geom_id])
                if rbound <= 0.0:
                    continue
                if np.dot(rel, rel) > (cutoff + rbound) ** 2:
                    continue

            loc_o = rel @ rot
            loc_d = directions @ rot
            if gtype == plane_t:
                # MuJoCo plane lies on local z=0, normal +z; ray hits only from z>0 going downward.
                if loc_o[2] <= 0.0:
                    continue
                dz = loc_d[:, 2]
                valid = dz < -eps
                if not np.any(valid):
                    continue
                t = np.full(len(directions), np.inf, dtype=np.float64)
                t[valid] = -loc_o[2] / dz[valid]
                hx = loc_o[0] + t * loc_d[:, 0]
                hy = loc_o[1] + t * loc_d[:, 1]
                hit = valid & (t >= 0.0) & (t <= cutoff)
                if size[0] > 0.0:
                    hit &= np.abs(hx) <= size[0]
                if size[1] > 0.0:
                    hit &= np.abs(hy) <= size[1]
            elif gtype == sphere_t:
                r = float(size[0])
                c = float(np.dot(loc_o, loc_o)) - r * r
                if c <= 0.0:
                    continue  # Origin inside sphere: MuJoCo returns miss
                b = loc_d @ loc_o
                disc = b * b - c
                valid = (b < 0.0) & (disc >= 0.0)
                if not np.any(valid):
                    continue
                t = np.full(len(directions), np.inf, dtype=np.float64)
                t[valid] = -b[valid] - np.sqrt(disc[valid])
                hit = valid & (t >= 0.0) & (t <= cutoff)
            elif gtype == box_t:
                if np.any(size <= 0.0):
                    continue
                if np.all(np.abs(loc_o) <= size):
                    continue  # Origin inside box
                # Slab intersection
                inv_d = np.where(np.abs(loc_d) > eps, 1.0 / np.where(np.abs(loc_d) > eps, loc_d, 1.0), np.inf)
                t1 = (-size - loc_o) * inv_d
                t2 = (size - loc_o) * inv_d
                # For parallel axes (|d| <= eps), check if origin is inside slab
                parallel = np.abs(loc_d) <= eps
                outside_parallel = parallel & (np.abs(loc_o) > size)
                tmin_axis = np.where(parallel, -np.inf, np.minimum(t1, t2))
                tmax_axis = np.where(parallel, np.inf, np.maximum(t1, t2))
                t_enter = np.max(tmin_axis, axis=1)
                t_exit = np.min(tmax_axis, axis=1)
                hit = (~np.any(outside_parallel, axis=1)) & (t_exit >= t_enter) & (t_enter >= 0.0) & (t_enter <= cutoff)
                t = t_enter
            elif gtype == cyl_t:
                r = float(size[0])
                hz = float(size[1])
                if r <= 0.0 or hz <= 0.0:
                    continue
                if (loc_o[0] * loc_o[0] + loc_o[1] * loc_o[1] <= r * r) and (abs(loc_o[2]) <= hz):
                    continue  # Origin inside cylinder
                t = np.full(len(directions), np.inf, dtype=np.float64)
                # Side surface: (ox + t dx)^2 + (oy + t dy)^2 = r^2
                a = loc_d[:, 0] ** 2 + loc_d[:, 1] ** 2
                b = loc_o[0] * loc_d[:, 0] + loc_o[1] * loc_d[:, 1]
                c = loc_o[0] ** 2 + loc_o[1] ** 2 - r * r
                disc = b * b - a * c
                side_ok = (a > eps) & (disc >= 0.0)
                if np.any(side_ok):
                    t_side = (-b[side_ok] - np.sqrt(disc[side_ok])) / a[side_ok]
                    z_side = loc_o[2] + t_side * loc_d[side_ok, 2]
                    good_side = (t_side >= 0.0) & (t_side <= cutoff) & (np.abs(z_side) <= hz)
                    idx = np.flatnonzero(side_ok)[good_side]
                    t[idx] = np.minimum(t[idx], t_side[good_side])
                # End caps z = +hz and z = -hz
                dz = loc_d[:, 2]
                cap_ok = np.abs(dz) > eps
                if np.any(cap_ok):
                    for zcap in (-hz, hz):
                        tc = (zcap - loc_o[2]) / dz[cap_ok]
                        xc = loc_o[0] + tc * loc_d[cap_ok, 0]
                        yc = loc_o[1] + tc * loc_d[cap_ok, 1]
                        good_cap = (tc >= 0.0) & (tc <= cutoff) & (xc * xc + yc * yc <= r * r)
                        idx = np.flatnonzero(cap_ok)[good_cap]
                        t[idx] = np.minimum(t[idx], tc[good_cap])
                hit = np.isfinite(t) & (t >= 0.0) & (t <= cutoff)
            else:
                continue

            closer = hit & ((geom_ids == -1) | (t < distances))
            if np.any(closer):
                distances[closer] = t[closer]
                geom_ids[closer] = geom_id

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
        if self._analytic_geoms is not None:
            _geomid.fill(-1)
            self._dist.fill(self.cutoff_dist)
            if self._analytic_geoms:
                self._trace_analytic_geoms(pnt.ravel(), world_vecs, _geomid, self._dist)
        else:
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
