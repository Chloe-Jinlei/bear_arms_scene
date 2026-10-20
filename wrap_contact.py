"""Scene-local sampled cloth/soft-bear contacts for Genesis 1.4.3.

Install after adding entities and before ``scene.build()``. Refresh the contact
cache from the current particle positions before each control/physics step.
Constraints operate on the actual deforming tetrahedral boundary. They move
both free bodies according to inverse mass, leave pinned particles fixed, and
never prescribe an object pose. This is discrete sampled surface contact, not
continuous collision detection or a guarantee of zero triangle intersections.
"""

from importlib.metadata import version
import time

import numpy as np
import trimesh


def tetra_boundary(points, tetra):
    """Return used particle indices and consistently outward boundary faces."""
    tetra = np.asarray(tetra, dtype=np.int32)
    faces = np.concatenate([tetra[:, [1, 2, 3]], tetra[:, [0, 3, 2]],
                            tetra[:, [0, 1, 3]], tetra[:, [0, 2, 1]]])
    opposite = np.concatenate([tetra[:, i] for i in range(4)])
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Bear tetrahedra contain a non-manifold face")
    exterior = counts[inverse] == 1
    faces, opposite = faces[exterior].copy(), opposite[exterior]
    triangles = points[faces]
    inward = np.einsum("ij,ij->i", np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0]), points[opposite] - triangles[:, 0]) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    used, inverse = np.unique(faces, return_inverse=True)
    return used, inverse.reshape(-1, 3)


def unique_edges(elements):
    elements = np.asarray(elements)
    return np.unique(np.sort(np.concatenate([elements[:, [i, j]]
                     for i in range(elements.shape[1])
                     for j in range(i + 1, elements.shape[1])]), axis=1), axis=0)


def winding_number(points, vertices, faces, chunk_size=32):
    """Solid-angle winding with bounded temporary memory; closed meshes only."""
    triangles = vertices[faces]
    result = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk_size):
        batch = points[start:start + chunk_size]
        a, b, c = (triangles[None, :, i] - batch[:, None] for i in range(3))
        la, lb, lc = (np.linalg.norm(value, axis=-1) for value in (a, b, c))
        numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
        denominator = la * lb * lc + np.einsum("...i,...i->...", a, b) * lc
        denominator += np.einsum("...i,...i->...", b, c) * la
        denominator += np.einsum("...i,...i->...", c, a) * lb
        result[start:start + len(batch)] = np.sum(2 * np.arctan2(numerator, denominator), axis=1) / (4 * np.pi)
    return result


class ContactGeometry:
    """Shared geometry and independent penetration measurements."""

    def __init__(self, bear_points, bear_tetra, cloth_points, cloth_faces,
                 margin=.002, broadphase=.012):
        self.used, self.body_faces = tetra_boundary(bear_points, bear_tetra)
        self.cloth_faces = np.asarray(cloth_faces, dtype=np.int32)
        self.cloth_edges = unique_edges(self.cloth_faces)
        self.margin, self.broadphase = float(margin), float(broadphase)
        self.entry_history = {}
        vertices = np.arange(len(cloth_points))
        self.sample_nodes = np.concatenate([
            np.repeat(vertices[:, None], 3, axis=1),
            np.column_stack([self.cloth_edges, self.cloth_edges[:, 1]]), self.cloth_faces])
        self.sample_weights = np.concatenate([
            np.tile([1., 0., 0.], (len(vertices), 1)),
            np.tile([.5, .5, 0.], (len(self.cloth_edges), 1)),
            np.full((len(self.cloth_faces), 3), 1 / 3)])
        self.sample_slices = {
            "vertices": slice(0, len(vertices)),
            "edge_midpoints": slice(len(vertices), len(vertices) + len(self.cloth_edges)),
            "face_centroids": slice(len(vertices) + len(self.cloth_edges), None),
        }
        mesh = self.mesh(bear_points)
        if not mesh.is_watertight or mesh.volume <= 0:
            raise ValueError("Bear contact boundary must be watertight and outward oriented")

    def mesh(self, bear):
        return trimesh.Trimesh(np.asarray(bear)[self.used], self.body_faces, process=False)

    def samples(self, cloth):
        return (np.asarray(cloth)[self.sample_nodes] * self.sample_weights[..., None]).sum(1)

    def query(self, bear, cloth):
        """Cache local samples against exact nearest current boundary triangles."""
        mesh, samples = self.mesh(bear), self.samples(cloth)
        selected = np.flatnonzero(((samples >= mesh.bounds[0] - self.broadphase)
                                & (samples <= mesh.bounds[1] + self.broadphase)).all(1))
        if not len(selected):
            self.entry_history.clear()
            return selected, None, None, None
        closest, distance, triangles = trimesh.proximity.closest_point(mesh, samples[selected])
        weights = trimesh.triangles.points_to_barycentric(mesh.triangles[triangles], closest)
        weights = np.clip(weights, 0., 1.)
        weights /= weights.sum(1, keepdims=True)
        normals = (mesh.vertex_normals[mesh.faces[triangles]] * weights[..., None]).sum(1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        signed = np.einsum("ij,ij->i", samples[selected] - closest, normals)
        valid = (distance < self.broadphase) | (signed < 0)
        inside = signed[valid] < -1e-7
        selected, triangles, weights, normals = (v[valid] for v in (selected, triangles, weights, normals))
        # Express the smooth outward normal in each current face's local frame.
        # The kernel reconstructs that frame from the moving bear every pass.
        tri = mesh.triangles[triangles]
        tangent = tri[:, 1] - tri[:, 0]
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        normal = mesh.face_normals[triangles]
        bitangent = np.cross(normal, tangent)
        local_normals = np.stack([np.einsum("ij,ij->i", normals, value)
                                 for value in (tangent, bitangent, normal)], axis=1)
        # Preserve the last exterior contact side if a discrete step enters a
        # thin limb. Choosing a new nearest face inside the limb can otherwise
        # flip the normal and push cloth through its opposite side.
        active=set(map(int,selected))
        self.entry_history={k:v for k,v in self.entry_history.items() if k in active}
        for k,sample in enumerate(selected):
            key=int(sample)
            if inside[k] and key in self.entry_history:
                face_id,bary,local_normal=self.entry_history[key]
                old_point=(mesh.triangles[face_id]*bary[:,None]).sum(0)
                new_point=(mesh.triangles[triangles[k]]*weights[k,:,None]).sum(0)
                # A history plane on the torso must not mask a new collision
                # with a neighboring arm. Retain only a local entry surface.
                if np.linalg.norm(old_point-new_point)<.015:
                    triangles[k]=face_id;weights[k]=bary;local_normals[k]=local_normal
            elif not inside[k]:
                self.entry_history[key]=(int(triangles[k]),weights[k].copy(),local_normals[k].copy())
        return selected, self.used[self.body_faces[triangles]], weights, local_normals

    def validate(self, bear, cloth, full=True, tolerance=.0005):
        """Measure penetration independently of cached solver normals.

        ``full=True`` uses closed-mesh solid-angle winding for inside/outside;
        the cheaper mode uses nearest smooth normals and is only diagnostic.
        Vertex, midpoint and centroid samples do not exhaust every triangle.
        """
        started = time.monotonic()
        mesh, samples = self.mesh(bear), self.samples(cloth)
        if not np.isfinite(samples).all() or not np.isfinite(mesh.vertices).all():
            raise ValueError("Non-finite deformable state")
        selected = np.flatnonzero(((samples >= mesh.bounds[0] - self.margin)
                                & (samples <= mesh.bounds[1] + self.margin)).all(1))
        signed = np.full(len(samples), np.inf)
        if len(selected):
            closest, distance, triangles = trimesh.proximity.closest_point(mesh, samples[selected])
            if full:
                inside = np.abs(winding_number(samples[selected], mesh.vertices, mesh.faces)) > .5
            else:
                barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[triangles], closest)
                normals = (mesh.vertex_normals[mesh.faces[triangles]] * barycentric[..., None]).sum(1)
                inside = np.einsum("ij,ij->i", samples[selected] - closest, normals) < 0
            signed[selected] = np.where(inside, -distance, distance)
        result = {"method": "solid-angle winding and exact triangle distance" if full else "nearest smooth-normal sign (diagnostic)",
                  "body_surface_vertices": len(mesh.vertices), "body_surface_triangles": len(mesh.faces),
                  "body_watertight": bool(mesh.is_watertight), "sample_count": len(samples),
                  "penetration_tolerance_m": tolerance,
                  "inside_samples": int(np.count_nonzero(signed < -tolerance)),
                  "max_penetration_m": float(max(0., -signed.min())),
                  "minimum_sample_clearance_m": float(signed.min()) if len(selected) else None,
                  "near_surface_samples": int(np.count_nonzero(np.abs(signed) < .006)),
                  "samples": {}}
        for name, section in self.sample_slices.items():
            values = signed[section]
            result["samples"][name] = {"count": len(values),
                "inside": int(np.count_nonzero(values < -tolerance)),
                "max_penetration_m": float(max(0., -values.min()))}
        result["no_sample_penetration_over_tolerance"] = result["inside_samples"] == 0
        result["elapsed_seconds"] = time.monotonic() - started
        return result


def install_surface_contacts(scene, bear, cloth, bear_points, bear_tetra,
                             cloth_points, cloth_faces, margin=.002,
                             iterations=4, max_correction=.0003, broadphase=.012,
                             friction=0., body_response_scale=1.):
    """Install reciprocal substep contacts and return their cache/controller.

    ``friction`` is a shared static/kinetic Coulomb coefficient. The default
    preserves the previous normal-only contact. A caller may use the smaller
    cloth/body coefficient, as the scene's rigid/PBD adapter does (the native
    PBD particle-pair solver instead uses the larger coefficient). Friction
    opposes relative tangential motion during this substep, with a budget
    bounded by the actual compressive normal correction. No contact history
    anchors the cloth to a world position or to a material point on the bear.
    """
    import genesis as gs
    import quadrants as qd
    import torch

    if version("genesis-world") != "1.4.3":
        raise RuntimeError("Review the private PBD hook before using another Genesis version")
    if scene.is_built:
        raise RuntimeError("Install surface contacts before scene.build()")
    if bear.solver is not cloth.solver:
        raise ValueError("Both deformables must belong to the same PBD solver")
    if not np.isfinite(friction) or friction < 0:
        raise ValueError("Surface contact friction must be finite and nonnegative")
    geometry = ContactGeometry(bear_points, bear_tetra, cloth_points, cloth_faces, margin, broadphase)

    @qd.data_oriented
    class SurfaceContacts:
        def __init__(self):
            self.geometry, self.solver = geometry, bear.solver
            self.native_volume = self.solver._kernel_solve_volume
            self.cache, self.count = None, 0
            self.iterations, self.margin = int(iterations), float(margin)
            self.friction = float(friction)
            self.body_response_scale = float(body_response_scale)
            self.refresh_seconds = 0.
            self.max_cached_constraints = 0
            self.limits = np.full(self.solver.n_particles, max_correction)
            incident=np.bincount((np.asarray(bear_tetra)+bear.particle_start).ravel(),minlength=self.solver.n_particles)
            self.volume_incidence=torch.as_tensor(np.maximum(incident,1).astype(np.int32),device=gs.device,dtype=gs.tc_int)
            for entity, points, elements in ((bear, bear_points, bear_tetra),
                                            (cloth, cloth_points, cloth_faces)):
                edges = unique_edges(elements)
                rest = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
                shortest = np.full(entity.n_particles, np.inf)
                np.minimum.at(shortest, edges[:, 0], rest)
                np.minimum.at(shortest, edges[:, 1], rest)
                self.limits[entity.particle_start:entity.particle_end] = np.minimum(max_correction, .1 * shortest)
            self.solver._kernel_solve_volume = self.solve
            self.description = {
                "method": "mass-weighted cloth vertex/mid-edge/face contacts against the moving tetrahedral bear boundary",
                "clearance_m": margin, "iterations_per_substep": iterations,
                "broadphase_distance_m": broadphase, "maximum_particle_correction_per_pass_m": max_correction,
                "equal_opposite_free_body_corrections": body_response_scale==1., "pinned_particles_unchanged": True,
                "custom_contact_body_inverse_mass_scale":body_response_scale,
                "coupling_note":"Additional sampled surface barrier uses the currently deforming body as an obstacle when scale=0; native particle collisions and robot/body dynamics remain active.",
                "body_surface_vertices": len(geometry.used), "body_surface_faces": len(geometry.body_faces),
                "cloth_contact_samples": len(geometry.sample_nodes),
                "normal_update": "cached smooth normal transported with current triangle frame each substep",
                "entry_side_history":"Retain the last exterior triangle and transported normal while a sample is signed-inside; discrete anti-crossing aid, not CCD.",
                "friction": {"coefficient": self.friction,
                    "model": "Coulomb static/kinetic position friction bounded by this pass's compressive normal multiplier",
                    "motion_reference": "relative cloth/body displacement from this substep's ipos",
                    "separated_contact_friction": False, "world_or_material_point_anchor": False,
                    "combined_normal_tangent_correction_limited": True},
                "object_pose_targets": False, "continuous_collision_detection": False,
                "signed_volume_barrier":{"target_minimum_rest_volume_fraction":.1,"relaxation":.7,"passes_per_volume_solve":3,
                    "maximum_particle_correction_per_pass_m":.00025,
                    "model":"Mass-weighted oriented-tetrahedron inequality constraint; fixed and link-attached nodes stay fixed. Applied only below the local volume floor; no target object pose."},
            }

        def refresh(self, body_positions=None, cloth_positions=None):
            started = time.monotonic()
            if body_positions is None:
                body_positions = bear.get_particles_pos().detach().cpu().numpy()
            if cloth_positions is None:
                cloth_positions = cloth.get_particles_pos().detach().cpu().numpy()
            selected, bn, bw, normals = geometry.query(body_positions, cloth_positions)
            self.count = len(selected)
            self.max_cached_constraints = max(self.max_cached_constraints, self.count)
            if self.count:
                cn = geometry.sample_nodes[selected] + cloth.particle_start
                cw = geometry.sample_weights[selected]
                bn = bn + bear.particle_start
                counts = np.bincount(np.concatenate([cn[cw > 1e-10], bn[bw > 1e-10]]),
                                     minlength=self.solver.n_particles)
                values = [cn, cw, bn, bw, normals, self.limits, counts]
                self.cache = [torch.as_tensor(np.ascontiguousarray(value), device=gs.device,
                              dtype=gs.tc_int if i in (0, 2, 6) else gs.tc_float)
                              for i, value in enumerate(values)]
            else:
                self.cache = None
            self.refresh_seconds += time.monotonic() - started
            return self.count

        def solve(self, f):
            self.native_volume(f)
            for _ in range(3):self.project_positive_volumes(self.volume_incidence)
            self.solver._kernel_solve_stretch(f)
            if self.cache is not None:
                for _ in range(self.iterations):
                    self.project(*self.cache)
                    self.native_volume(f)
                    for _ in range(3):self.project_positive_volumes(self.volume_incidence)
                self.project(*self.cache)

        @qd.kernel
        def project_positive_volumes(self, counts: qd.types.ndarray()):
            for i,b in qd.ndrange(self.solver.n_particles,self.solver._B):
                self.solver.particles[i,b].dpos.fill(0.)
            for el,b in qd.ndrange(self.solver._n_elems,self.solver._B):
                ids=qd.Vector([self.solver.elems_info[el].v1,self.solver.elems_info[el].v2,
                    self.solver.elems_info[el].v3,self.solver.elems_info[el].v4])
                p1=self.solver.particles[ids[0],b].pos
                p2=self.solver.particles[ids[1],b].pos
                p3=self.solver.particles[ids[2],b].pos
                p4=self.solver.particles[ids[3],b].pos
                volume=(p2-p1).dot((p3-p1).cross(p4-p1))/6.
                constraint=volume-.1*self.solver.elems_info[el].vol_rest
                if constraint<0.:
                    gradients=qd.Matrix.rows([(p4-p2).cross(p3-p2)/6.,(p3-p1).cross(p4-p1)/6.,
                        (p4-p1).cross(p2-p1)/6.,(p2-p1).cross(p3-p1)/6.])
                    denominator=gs.qd_float(0.)
                    for j in qd.static(range(4)):
                        inverse_mass=self.solver.particles[ids[j],b].free/self.solver.particles_info[ids[j]].mass
                        denominator+=inverse_mass*gradients[j,:].norm_sqr()
                    multiplier=-.7*constraint/qd.max(denominator,1e-20)
                    for j in qd.static(range(4)):
                        inverse_mass=self.solver.particles[ids[j],b].free/self.solver.particles_info[ids[j]].mass
                        self.solver.particles[ids[j],b].dpos+=multiplier*inverse_mass*gradients[j,:]
            for i,b in qd.ndrange(self.solver.n_particles,self.solver._B):
                if self.solver.particles[i,b].free:
                    delta=self.solver.particles[i,b].dpos/counts[i]
                    delta*=qd.min(1.,.00025/qd.max(delta.norm(),1e-20))
                    self.solver.particles[i,b].pos+=delta
                self.solver.particles[i,b].dpos.fill(0.)

        @qd.kernel
        def project(self, cn: qd.types.ndarray(), cw: qd.types.ndarray(),
                    bn: qd.types.ndarray(), bw: qd.types.ndarray(),
                    normals: qd.types.ndarray(), limits: qd.types.ndarray(), counts: qd.types.ndarray()):
            for k in range(cn.shape[0]):
                cpos = qd.Vector.zero(gs.qd_float, 3)
                bpos = qd.Vector.zero(gs.qd_float, 3)
                cprevious = qd.Vector.zero(gs.qd_float, 3)
                bprevious = qd.Vector.zero(gs.qd_float, 3)
                denominator = gs.qd_float(0.)
                max_multiplier = gs.qd_float(1e9)
                a = self.solver.particles[bn[k, 0], 0].pos
                b = self.solver.particles[bn[k, 1], 0].pos
                c = self.solver.particles[bn[k, 2], 0].pos
                tangent = (b - a).normalized(1e-12)
                face_normal = (b - a).cross(c - a).normalized(1e-12)
                bitangent = face_normal.cross(tangent)
                normal = (normals[k, 0] * tangent + normals[k, 1] * bitangent
                          + normals[k, 2] * face_normal).normalized(1e-12)
                for j in qd.static(range(3)):
                    ci, bi = cn[k, j], bn[k, j]
                    cweight = self.solver.particles[ci, 0].free / self.solver.particles_info[ci].mass
                    bweight = self.body_response_scale * self.solver.particles[bi, 0].free / self.solver.particles_info[bi].mass
                    cpos += cw[k, j] * self.solver.particles[ci, 0].pos
                    bpos += bw[k, j] * self.solver.particles[bi, 0].pos
                    cprevious += cw[k, j] * self.solver.particles[ci, 0].ipos
                    bprevious += bw[k, j] * self.solver.particles[bi, 0].ipos
                    denominator += cweight * cw[k, j] ** 2 + bweight * bw[k, j] ** 2
                    if cweight * cw[k, j] > 1e-12:
                        max_multiplier = qd.min(max_multiplier, limits[ci] / (cweight * cw[k, j] * qd.max(counts[ci], 1)))
                    if bweight * bw[k, j] > 1e-12:
                        max_multiplier = qd.min(max_multiplier, limits[bi] / (bweight * bw[k, j] * qd.max(counts[bi], 1)))
                violation = self.margin - (cpos - bpos).dot(normal)
                if violation > 0 and denominator > 1e-12:
                    multiplier = qd.min(violation / denominator, max_multiplier)
                    impulse = multiplier * normal
                    if qd.static(self.friction > 0):
                        relative_delta = (cpos - cprevious) - (bpos - bprevious)
                        tangential_delta = relative_delta - normal * relative_delta.dot(normal)
                        tangential_length = tangential_delta.norm()
                        if tangential_length > 1e-12:
                            # Static friction removes all resolvable relative
                            # slip; sliding friction is limited by mu*lambda_N.
                            tangent_multiplier = qd.min(tangential_length / denominator,
                                                        self.friction * multiplier)
                            impulse -= tangent_multiplier * tangential_delta / tangential_length
                        # Share the same per-particle correction cap between
                        # normal and tangential impulses. Scaling both retains
                        # |lambda_T| <= mu * lambda_N after limiting.
                        impulse *= qd.min(1., max_multiplier / qd.max(impulse.norm(), 1e-30))
                    for j in qd.static(range(3)):
                        ci, bi = cn[k, j], bn[k, j]
                        cweight = self.solver.particles[ci, 0].free / self.solver.particles_info[ci].mass
                        bweight = self.body_response_scale * self.solver.particles[bi, 0].free / self.solver.particles_info[bi].mass
                        self.solver.particles[ci, 0].dpos += cweight * cw[k, j] * impulse
                        self.solver.particles[bi, 0].dpos -= bweight * bw[k, j] * impulse
            for i in range(self.solver.n_particles):
                if self.solver.particles[i, 0].free:
                    self.solver.particles[i, 0].pos += self.solver.particles[i, 0].dpos
                self.solver.particles[i, 0].dpos.fill(0.)

        def validate(self, bear_positions, cloth_positions, full=True, tolerance=.0005):
            return geometry.validate(bear_positions, cloth_positions, full=full, tolerance=tolerance)

        def close(self):
            self.solver._kernel_solve_volume = self.native_volume

    return SurfaceContacts()
