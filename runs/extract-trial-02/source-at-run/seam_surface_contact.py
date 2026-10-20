"""Unsigned sampled VF plus finite edge-edge cloth contact for Genesis 1.4.3.

Install after bear surface contact and before velcro adhesion, before build.
Call refresh() before every scene.step(). Both regions are free cloth: contact
moves their nodes reciprocally using inverse mass and squared barycentric
weights. No attraction, adhesion, friction, normal alignment or world anchors.

Only actual finite sample-to-triangle or segment-to-segment distance below the thickness generates
repulsion. This applies to face interiors, internal mesh edges, finite patch
boundaries and vertices alike. Side history is used only to select a normal
at nearly zero distance; it never marks an open sheet's far side as penetrated.

This is discrete sampled contact, not CCD: an already crossed configuration
is repelled toward its present side. Nearest triangles are refreshed each
control step; actual triangle geometry and closest barycentrics are recomputed
each projection. Restore from nonintersecting geometry. Transient cache history
need not persist; only an exactly coincident restore has an ambiguous normal.
"""
from importlib.metadata import version
import time

import numpy as np
import trimesh


DEFAULTS = {
    "region_width_m": .10,
    "thickness_m": .002,
    "broadphase_m": .012,
    "history_release_m": .020,
    "iterations_per_substep": 3,
    "max_correction_per_pass_m": .0003,
    "relaxation": .8,
}


def _normals(triangles):
    n = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(n, axis=1)
    return n / np.maximum(length[:, None], 1e-15), length


def _samples(nodes, faces):
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]]]), axis=1), axis=0)
    indices = np.concatenate([np.repeat(nodes[:, None], 3, axis=1),
                              np.column_stack([edges, edges[:, 1]]), faces])
    weights = np.concatenate([np.tile([1., 0., 0.], (len(nodes), 1)),
                              np.tile([.5, .5, 0.], (len(edges), 1)),
                              np.full((len(faces), 3), 1 / 3)])
    return indices, weights, {"vertices": len(nodes), "edge_midpoints": len(edges),
                               "face_centroids": len(faces)}


def _distance_contact(samples, triangles, fallback_sides):
    """Finite closest-point distance; history only resolves coincidence."""
    closest = trimesh.triangles.closest_point(triangles, samples)
    bary = trimesh.triangles.points_to_barycentric(triangles, closest)
    bary = np.clip(bary, 0., 1.)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-15)
    normal, area2 = _normals(triangles)
    normal *= fallback_sides[:, None]
    delta = samples - closest
    distance = np.linalg.norm(delta, axis=1)
    nonzero = distance > 1e-9
    normal[nonzero] = delta[nonzero] / distance[nonzero, None]
    return closest, bary, normal, distance, area2


def _closest_segments(a, b, c, d):
    """Closest finite segment parameters; broadcasting arrays end in XYZ.

    Near-parallel pairs use the endpoint branch to avoid cancellation. Both
    zero-length segments and clamped endpoint features remain well defined.
    """
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = (np.sum(x * y, axis=-1) for x, y in ((u, u), (u, v), (v, v)))
    dd, ee = np.sum(u * w, axis=-1), np.sum(v * w, axis=-1)
    aa, bb, cc, dd, ee = np.broadcast_arrays(aa, bb, cc, dd, ee)
    denominator = aa * cc - bb * bb
    s = np.zeros_like(denominator)
    nonparallel = denominator > 1e-7 * aa * cc
    np.divide(bb * ee - cc * dd, denominator, out=s, where=nonparallel)
    s = np.clip(s, 0., 1.)
    t = np.divide(bb * s + ee, cc, out=np.zeros_like(s), where=cc > 1e-20)
    low, high = t < 0., t > 1.
    s = np.where(low, np.divide(-dd, aa, out=np.zeros_like(s), where=aa > 1e-20), s)
    s = np.where(high, np.divide(bb - dd, aa, out=np.zeros_like(s), where=aa > 1e-20), s)
    # A degenerate target becomes a point on the source segment.
    s = np.where(cc <= 1e-20, np.divide(-dd, aa, out=np.zeros_like(s), where=aa > 1e-20), s)
    s, t = np.clip(s, 0., 1.), np.clip(t, 0., 1.)
    return s, t, a + s[..., None] * u, c + t[..., None] * v


class SeamContactGeometry:
    """Pure geometric candidate cache; all node indices are local to cloth."""
    def __init__(self, points, faces, config=None):
        supplied = config or {}
        supplied = supplied.get("seam_surface_contact", supplied)
        self.config = {**DEFAULTS, **supplied}
        self.rest = np.asarray(points, dtype=float).copy()
        self.faces = np.asarray(faces, dtype=np.int32).copy()
        width = float(self.config["region_width_m"])
        h, broad, release = (float(self.config[k]) for k in
                              ("thickness_m", "broadphase_m", "history_release_m"))
        values = [width, h, broad, release, self.config["max_correction_per_pass_m"],
                  self.config["relaxation"]]
        if not (np.isfinite(values).all() and 0 < width < np.ptp(self.rest[:, 0]) / 2
                and 0 < h < broad <= release
                and 0 < self.config["max_correction_per_pass_m"] <= h
                and 0 < self.config["relaxation"] <= 1
                and int(self.config["iterations_per_substep"]) == self.config["iterations_per_substep"]
                and self.config["iterations_per_substep"] >= 1):
            raise ValueError("Invalid seam contact region, clearance, search distance or projection parameters")
        self.regions = {
            "left": np.flatnonzero(self.rest[:, 0] <= self.rest[:, 0].min() + width + 1e-9),
            "right": np.flatnonzero(self.rest[:, 0] >= self.rest[:, 0].max() - width - 1e-9),
        }
        self.region_faces = {s: np.flatnonzero(np.isin(self.faces, ids).all(1))
                             for s, ids in self.regions.items()}
        if any(not len(ids) for ids in self.region_faces.values()):
            raise ValueError("Both seam regions must contain complete triangles")
        self.region_edges = {}
        for side, ids in self.region_faces.items():
            f = self.faces[ids]
            self.region_edges[side] = np.unique(np.sort(np.concatenate(
                [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
        self.sample_nodes, self.sample_weights, self.directions = [], [], []
        self.sample_counts, start = {}, 0
        for side, other in (("left", "right"), ("right", "left")):
            ids, weights, counts = _samples(self.regions[side], self.faces[self.region_faces[side]])
            self.sample_nodes.append(ids)
            self.sample_weights.append(weights)
            self.sample_counts[side] = counts
            self.directions.append((side, np.arange(start, start + len(ids)), self.region_faces[other]))
            start += len(ids)
        self.sample_nodes = np.concatenate(self.sample_nodes).astype(np.int32)
        self.sample_weights = np.concatenate(self.sample_weights)
        self.clear_history()

    def clear_history(self):
        self.history = {}
        self.edge_history = {}
        self.previous_positions = None
        self.initial_side_ambiguities = 0
        self.initial_edge_normal_ambiguities = 0

    def _query_edges(self, p):
        left, right = self.region_edges["left"], self.region_edges["right"]
        a, b = p[left], p[right]
        broad = self.config["broadphase_m"]
        i, j = np.where(np.all(a.max(1)[:, None] + broad >= b.min(1)[None], axis=2)
                        & np.all(b.max(1)[None] + broad >= a.min(1)[:, None], axis=2))
        _, _, pa, pb = _closest_segments(a[i, 0], a[i, 1], b[j, 0], b[j, 1])
        delta = pa - pb
        distance = np.linalg.norm(delta, axis=1)
        kept = distance <= broad
        i, j, delta, distance = i[kept], j[kept], delta[kept], distance[kept]
        normals = np.zeros_like(delta)
        nonzero = distance > 1e-9
        normals[nonzero] = delta[nonzero] / distance[nonzero, None]
        for k in np.flatnonzero(~nonzero):
            key = (int(i[k]), int(j[k]))
            old = self.edge_history.get(key)
            if old is None and self.previous_positions is not None:
                prev = self.previous_positions
                _, _, qa, qb = _closest_segments(prev[left[i[k], 0]], prev[left[i[k], 1]],
                                                 prev[right[j[k], 0]], prev[right[j[k], 1]])
                old = qa - qb
                if np.linalg.norm(old) <= 1e-9:
                    old = None
            if old is None:
                u, v = a[i[k], 1] - a[i[k], 0], b[j[k], 1] - b[j[k], 0]
                old = np.cross(u, v)
                if np.linalg.norm(old) <= 1e-12:
                    direction = u if np.linalg.norm(u) > 1e-12 else v
                    axis = np.eye(3)[np.argmin(abs(direction))]
                    old = np.cross(direction, axis)
                if np.linalg.norm(old) <= 1e-12:
                    old = np.array([1., 0., 0.])
                self.initial_edge_normal_ambiguities += 1
            normals[k] = old / np.linalg.norm(old)
        self.edge_history = {(int(x), int(y)): n.copy() for x, y, n in zip(i, j, normals)}
        return left[i], right[j], normals

    def samples(self, positions):
        return (np.asarray(positions)[self.sample_nodes] * self.sample_weights[..., None]).sum(1)

    def query(self, positions):
        p = np.asarray(positions, dtype=float)
        if p.shape != self.rest.shape or not np.isfinite(p).all():
            raise ValueError("Seam contact requires finite cloth positions with unchanged topology")
        sample_points = self.samples(p)
        previous_samples = self.samples(self.previous_positions) if self.previous_positions is not None else None
        selected_all, face_all, side_all, history = [], [], [], {}
        release, broad = self.config["history_release_m"], self.config["broadphase_m"]
        for side_name, sample_ids, face_ids in self.directions:
            tri = p[self.faces[face_ids]]
            bounds = np.array([tri.min((0, 1)), tri.max((0, 1))])
            selected = sample_ids[((sample_points[sample_ids] >= bounds[0] - release)
                                  & (sample_points[sample_ids] <= bounds[1] + release)).all(1)]
            if not len(selected):
                continue
            mesh = trimesh.Trimesh(p, self.faces[face_ids], process=False)
            closest, distance, local_face = trimesh.proximity.closest_point(mesh, sample_points[selected])
            kept = distance <= release
            selected, closest, distance = selected[kept], closest[kept], distance[kept]
            actual_faces = face_ids[local_face[kept]]
            current_tri = p[self.faces[actual_faces]]
            normal, area2 = _normals(current_tri)
            for j, sample_id in enumerate(selected):
                if area2[j] <= 1e-14:
                    continue
                sample_id, face_id = int(sample_id), int(actual_faces[j])
                signed = float(np.dot(sample_points[sample_id] - closest[j], normal[j]))
                old = self.history.get(sample_id)
                if distance[j] > 1e-9:
                    # Nonzero current distance always defines the real side.
                    # This sign is only a possible future coincidence fallback.
                    side = np.sign(signed)
                elif old is not None:
                    old_normal, _ = _normals(p[self.faces[[old[0]]]])
                    # At coincidence only, transport the most recent observed
                    # side with the moving target triangle.
                    side = 1. if np.dot(normal[j], old_normal[0] * old[1]) >= 0 else -1.
                elif self.previous_positions is not None:
                    old_tri = self.previous_positions[self.faces[face_id]]
                    old_normal, _ = _normals(old_tri[None])
                    old_signed = float(np.dot(previous_samples[sample_id] - old_tri[0], old_normal[0]))
                    side = np.sign(old_signed) if abs(old_signed) > 1e-9 else np.sign(signed)
                else:
                    side = np.sign(signed)
                if side == 0:
                    # Coincident initialization has no observable incoming side.
                    # Use a deterministic pair ordering; report this ambiguity.
                    side = 1. if side_name == "left" else -1.
                    self.initial_side_ambiguities += 1
                history[sample_id] = (face_id, float(side))
                if distance[j] <= broad:
                    selected_all.append(sample_id)
                    face_all.append(face_id)
                    side_all.append(side)
        edge_a, edge_b, edge_normals = self._query_edges(p)
        self.history, self.previous_positions = history, p.copy()
        selected = np.asarray(selected_all, dtype=np.int32)
        target_faces = np.asarray(face_all, dtype=np.int32)
        return {"selected": selected, "source_nodes": self.sample_nodes[selected],
                "source_weights": self.sample_weights[selected],
                "target_face_ids": target_faces, "target_nodes": self.faces[target_faces],
                "sides": np.asarray(side_all, dtype=float),
                "edge_a": edge_a, "edge_b": edge_b, "edge_normals": edge_normals}

    def _measure_vertex_face(self, positions, cache):
        common = {"crossed_samples": None, "minimum_oriented_gap_m": None,
                  "crossing_measurement": "not applicable: unsigned discrete contact is not a crossing or penetration test"}
        if cache is None or not len(cache["selected"]):
            return {**common, "cached_constraints": 0, "violating_samples": 0,
                    "max_clearance_violation_m": 0., "minimum_surface_distance_m": None}
        p = np.asarray(positions)
        samples = (p[cache["source_nodes"]] * cache["source_weights"][..., None]).sum(1)
        _, _, _, distance, area2 = _distance_contact(samples, p[cache["target_nodes"]], cache["sides"])
        valid = area2 > 1e-14
        return {**common, "cached_constraints": len(samples),
                "violating_samples": int(np.count_nonzero(valid & (distance < self.config["thickness_m"]))),
                "max_clearance_violation_m": float(max(0., np.max(self.config["thickness_m"] - distance[valid]))) if valid.any() else 0.,
                "minimum_surface_distance_m": float(distance[valid].min()) if valid.any() else None}

    def measure(self, positions, cache):
        result = self._measure_vertex_face(positions, cache)
        result["cached_vertex_face_constraints"] = result["cached_constraints"]
        edge_count = len(cache["edge_a"]) if cache is not None else 0
        result.update(cached_edge_edge_constraints=edge_count, violating_edge_pairs=0,
                      minimum_edge_edge_distance_m=None, max_edge_clearance_violation_m=0.)
        result["cached_constraints"] += edge_count
        if edge_count:
            p = np.asarray(positions)
            a, b = p[cache["edge_a"]], p[cache["edge_b"]]
            _, _, pa, pb = _closest_segments(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
            d = np.linalg.norm(pa - pb, axis=1)
            h = self.config["thickness_m"]
            result["violating_edge_pairs"] = int(np.count_nonzero(d < h))
            result["minimum_edge_edge_distance_m"] = float(d.min())
            result["max_edge_clearance_violation_m"] = float(max(0., h - d.min()))
            result["max_clearance_violation_m"] = max(result["max_clearance_violation_m"],
                                                       result["max_edge_clearance_violation_m"])
            previous = result["minimum_surface_distance_m"]
            result["minimum_surface_distance_m"] = float(d.min()) if previous is None else min(previous, float(d.min()))
        return result


def install(scene, cloth, points, faces, config=None):
    """Install after bear contact and before adhesion; refresh every scene step.

    close() must be called in reverse installation order. reset_cache() clears
    transient coincidence-normal history after a restored state; no body is changed.
    """
    import genesis as gs
    import quadrants as qd
    import torch

    if version("genesis-world") != "1.4.3" or scene.is_built:
        raise RuntimeError("Install this Genesis 1.4.3 seam contact adapter before scene.build()")
    geometry = SeamContactGeometry(points, faces, config)

    @qd.data_oriented
    class SeamContacts:
        def __init__(self):
            self.geometry, self.config, self.cloth = geometry, geometry.config, cloth
            self.solver = cloth.solver
            self.previous_solve = self.solver._kernel_solve_volume
            self.solver._kernel_solve_volume = self.solve
            self.cache, self.gpu_cache = None, None
            self.refresh_seconds, self.max_cached_constraints = 0., 0
            self.description = {
                "model": "reciprocal unsigned-distance sampled VF plus finite edge-edge contact between the two cloth end regions",
                "sample_counts": geometry.sample_counts,
                "edge_counts": {s: len(e) for s, e in geometry.region_edges.items()},
                "edge_pair_policy": "all close opposite-region edge pairs; closest segment parameters recomputed each substep",
                "shared_vf_ee_correction_cap": True,
                "participation_counting": "rebuilt every projection from actual active VF and EE distances; conservative target corners/endpoints",
                "region_width_m": self.config["region_width_m"],
                "thickness_m": self.config["thickness_m"],
                "broadphase_m": self.config["broadphase_m"],
                "history_release_m": self.config["history_release_m"],
                "maximum_particle_correction_per_pass_m": self.config["max_correction_per_pass_m"],
                "iterations_per_substep": self.config["iterations_per_substep"],
                "friction": False, "adhesion": False, "normal_alignment_constraint": False,
                "world_anchors": False, "reciprocal_inverse_mass_projection": True,
                "continuous_collision_detection": False,
                "same_region_contacts": False,
                "all_features": "unsigned radial repulsion only at actual finite triangle or segment distance below thickness",
                "history_usage": "normal selection only at distance <= 1e-9 m; no side or crossing constraint",
                "restoration": "transient cache rebuilt; resume from nonintersecting geometry; exact coincidence has ambiguous incoming normal",
            }

        def _positions(self):
            return cloth.get_particles_pos().detach().cpu().numpy()

        def refresh(self, cloth_positions=None, *, positions=None):
            started = time.monotonic()
            if cloth_positions is not None and positions is not None:
                raise ValueError("Supply only one positions argument")
            p = positions if positions is not None else cloth_positions
            p = self._positions() if p is None else p
            self.cache = geometry.query(p)
            count = len(self.cache["selected"]) + len(self.cache["edge_a"])
            self.max_cached_constraints = max(self.max_cached_constraints, count)
            if count:
                cn = self.cache["source_nodes"] + cloth.particle_start
                cw = self.cache["source_weights"]
                bn = self.cache["target_nodes"] + cloth.particle_start
                ea = self.cache["edge_a"] + cloth.particle_start
                eb = self.cache["edge_b"] + cloth.particle_start
                # This workspace is rebuilt on the GPU each projection. Far
                # broadphase pairs must not dilute the active contact budget.
                counts = np.zeros(self.solver.n_particles, dtype=np.int32)
                arrays = [cn, cw, bn, self.cache["sides"], ea, eb, self.cache["edge_normals"], counts]
                self.gpu_cache = [torch.as_tensor(np.ascontiguousarray(v), device=gs.device,
                                  dtype=gs.tc_int if i in (0, 2, 4, 5, 7) else gs.tc_float)
                                  for i, v in enumerate(arrays)]
            else:
                self.gpu_cache = None
            self.refresh_seconds += time.monotonic() - started
            return count

        @qd.func
        def _closest_bary(self, p, a, b, c):
            # Ericson's point/triangle Voronoi-region construction.
            ab, ac, ap = b - a, c - a, p - a
            d1, d2 = ab.dot(ap), ac.dot(ap)
            bp, cp = p - b, p - c
            d3, d4, d5, d6 = ab.dot(bp), ac.dot(bp), ab.dot(cp), ac.dot(cp)
            vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
            bary = qd.Vector([0., 0., 0.])
            if d1 <= 0 and d2 <= 0:
                bary = qd.Vector([1., 0., 0.])
            elif d3 >= 0 and d4 <= d3:
                bary = qd.Vector([0., 1., 0.])
            elif vc <= 0 and d1 >= 0 and d3 <= 0:
                v = d1 / qd.max(d1 - d3, 1e-20)
                bary = qd.Vector([1. - v, v, 0.])
            elif d6 >= 0 and d5 <= d6:
                bary = qd.Vector([0., 0., 1.])
            elif vb <= 0 and d2 >= 0 and d6 <= 0:
                w = d2 / qd.max(d2 - d6, 1e-20)
                bary = qd.Vector([1. - w, 0., w])
            elif va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
                w = (d4 - d3) / qd.max(d4 - d3 + d5 - d6, 1e-20)
                bary = qd.Vector([0., 1. - w, w])
            else:
                denom = qd.max(va + vb + vc, 1e-20)
                v, w = vb / denom, vc / denom
                bary = qd.Vector([1. - v - w, v, w])
            return bary

        @qd.func
        def _closest_segment_parameters(self, a, b, c, d):
            u, v, w = b - a, d - c, a - c
            aa, bb, cc, dd, ee = u.dot(u), u.dot(v), v.dot(v), u.dot(w), v.dot(w)
            denominator = aa * cc - bb * bb
            s, t = gs.qd_float(0.), gs.qd_float(0.)
            if denominator > 1e-7 * aa * cc:
                s = qd.min(1., qd.max(0., (bb * ee - cc * dd) / denominator))
            if cc > 1e-20:
                t = (bb * s + ee) / cc
                if t < 0.:
                    t = 0.
                    if aa > 1e-20:
                        s = qd.min(1., qd.max(0., -dd / aa))
                elif t > 1.:
                    t = 1.
                    if aa > 1e-20:
                        s = qd.min(1., qd.max(0., (bb - dd) / aa))
            elif aa > 1e-20:
                s = qd.min(1., qd.max(0., -dd / aa))
            return qd.Vector([s, t])

        @qd.kernel
        def project(self, cn: qd.types.ndarray(), cw: qd.types.ndarray(),
                    bn: qd.types.ndarray(), sides: qd.types.ndarray(),
                    ea: qd.types.ndarray(), eb: qd.types.ndarray(), en: qd.types.ndarray(),
                    counts: qd.types.ndarray()):
            for i in range(cloth.particle_start, cloth.particle_end):
                self.solver.particles[i, 0].dpos.fill(0.)
                counts[i] = 0
            # Count only constraints that are active at these unchanged
            # positions. Include every target corner/edge endpoint as a
            # conservative bound; no world or pinned node is moved.
            for k in range(cn.shape[0]):
                cpos = qd.Vector.zero(gs.qd_float, 3)
                for j in qd.static(range(3)):
                    cpos += cw[k, j] * self.solver.particles[cn[k, j], 0].pos
                a = self.solver.particles[bn[k, 0], 0].pos
                b = self.solver.particles[bn[k, 1], 0].pos
                c = self.solver.particles[bn[k, 2], 0].pos
                if (b - a).cross(c - a).norm() > 1e-14:
                    bw = self._closest_bary(cpos, a, b, c)
                    bpos = bw[0] * a + bw[1] * b + bw[2] * c
                    delta = cpos - bpos
                    distance = delta.norm()
                    if distance < self.config["thickness_m"]:
                        for j in qd.static(range(3)):
                            if cw[k, j] > 1e-12:
                                counts[cn[k, j]] += 1
                            counts[bn[k, j]] += 1
            for k in range(ea.shape[0]):
                a = self.solver.particles[ea[k, 0], 0].pos
                b = self.solver.particles[ea[k, 1], 0].pos
                c = self.solver.particles[eb[k, 0], 0].pos
                d = self.solver.particles[eb[k, 1], 0].pos
                st = self._closest_segment_parameters(a, b, c, d)
                aw, bw = qd.Vector([1. - st[0], st[0]]), qd.Vector([1. - st[1], st[1]])
                delta = aw[0] * a + aw[1] * b - bw[0] * c - bw[1] * d
                if delta.norm() < self.config["thickness_m"]:
                    for j in qd.static(range(2)):
                        counts[ea[k, j]] += 1
                        counts[eb[k, j]] += 1
            for k in range(cn.shape[0]):
                cpos = qd.Vector.zero(gs.qd_float, 3)
                for j in qd.static(range(3)):
                    cpos += cw[k, j] * self.solver.particles[cn[k, j], 0].pos
                a = self.solver.particles[bn[k, 0], 0].pos
                b = self.solver.particles[bn[k, 1], 0].pos
                c = self.solver.particles[bn[k, 2], 0].pos
                cross = (b - a).cross(c - a)
                if cross.norm() > 1e-14:
                    bw = self._closest_bary(cpos, a, b, c)
                    bpos = bw[0] * a + bw[1] * b + bw[2] * c
                    normal = sides[k] * cross.normalized(1e-15)
                    delta = cpos - bpos
                    distance = delta.norm()
                    if distance > 1e-9:
                        normal = delta / distance
                    violation = self.config["thickness_m"] - distance
                    if violation > 0:
                        denominator, limit = gs.qd_float(0.), gs.qd_float(1e9)
                        for j in qd.static(range(3)):
                            ci, bi = cn[k, j], bn[k, j]
                            wi = self.solver.particles[ci, 0].free / self.solver.particles_info[ci].mass
                            wj = self.solver.particles[bi, 0].free / self.solver.particles_info[bi].mass
                            denominator += wi * cw[k, j] ** 2 + wj * bw[j] ** 2
                            if wi * cw[k, j] > 1e-12:
                                limit = qd.min(limit, self.config["max_correction_per_pass_m"] /
                                               (wi * cw[k, j] * qd.max(counts[ci], 1)))
                            if wj * bw[j] > 1e-12:
                                limit = qd.min(limit, self.config["max_correction_per_pass_m"] /
                                               (wj * bw[j] * qd.max(counts[bi], 1)))
                        if denominator > 1e-12:
                            multiplier = qd.min(self.config["relaxation"] * violation / denominator, limit)
                            for j in qd.static(range(3)):
                                ci, bi = cn[k, j], bn[k, j]
                                wi = self.solver.particles[ci, 0].free / self.solver.particles_info[ci].mass
                                wj = self.solver.particles[bi, 0].free / self.solver.particles_info[bi].mass
                                self.solver.particles[ci, 0].dpos += wi * cw[k, j] * multiplier * normal
                                self.solver.particles[bi, 0].dpos -= wj * bw[j] * multiplier * normal
            # Accumulate EE before applying any VF or EE correction, preserving
            # reciprocal Jacobi updates and the common per-particle total cap.
            for k in range(ea.shape[0]):
                a = self.solver.particles[ea[k, 0], 0].pos
                b = self.solver.particles[ea[k, 1], 0].pos
                c = self.solver.particles[eb[k, 0], 0].pos
                d = self.solver.particles[eb[k, 1], 0].pos
                st = self._closest_segment_parameters(a, b, c, d)
                aw, bw = qd.Vector([1. - st[0], st[0]]), qd.Vector([1. - st[1], st[1]])
                delta = aw[0] * a + aw[1] * b - bw[0] * c - bw[1] * d
                distance = delta.norm()
                normal = qd.Vector([en[k, 0], en[k, 1], en[k, 2]])
                if distance > 1e-9:
                    normal = delta / distance
                violation = self.config["thickness_m"] - distance
                if violation > 0:
                    denominator, limit = gs.qd_float(0.), gs.qd_float(1e9)
                    for j in qd.static(range(2)):
                        ai, bi = ea[k, j], eb[k, j]
                        wi = self.solver.particles[ai, 0].free / self.solver.particles_info[ai].mass
                        wj = self.solver.particles[bi, 0].free / self.solver.particles_info[bi].mass
                        denominator += wi * aw[j] ** 2 + wj * bw[j] ** 2
                        if wi * aw[j] > 1e-12:
                            limit = qd.min(limit, self.config["max_correction_per_pass_m"] /
                                           (wi * aw[j] * qd.max(counts[ai], 1)))
                        if wj * bw[j] > 1e-12:
                            limit = qd.min(limit, self.config["max_correction_per_pass_m"] /
                                           (wj * bw[j] * qd.max(counts[bi], 1)))
                    if denominator > 1e-12:
                        multiplier = qd.min(self.config["relaxation"] * violation / denominator, limit)
                        for j in qd.static(range(2)):
                            ai, bi = ea[k, j], eb[k, j]
                            wi = self.solver.particles[ai, 0].free / self.solver.particles_info[ai].mass
                            wj = self.solver.particles[bi, 0].free / self.solver.particles_info[bi].mass
                            self.solver.particles[ai, 0].dpos += wi * aw[j] * multiplier * normal
                            self.solver.particles[bi, 0].dpos -= wj * bw[j] * multiplier * normal
            for i in range(cloth.particle_start, cloth.particle_end):
                if self.solver.particles[i, 0].free:
                    self.solver.particles[i, 0].pos += self.solver.particles[i, 0].dpos
                self.solver.particles[i, 0].dpos.fill(0.)

        def solve(self, f):
            self.previous_solve(f)
            if self.gpu_cache is not None:
                for _ in range(int(self.config["iterations_per_substep"])):
                    self.project(*self.gpu_cache)

        def metrics(self, positions=None):
            p = self._positions() if positions is None else positions
            return {**geometry.measure(p, self.cache),
                    "maximum_cached_constraints": self.max_cached_constraints,
                    "refresh_seconds": self.refresh_seconds,
                    "retained_side_samples": len(geometry.history),
                    "initial_side_ambiguities": geometry.initial_side_ambiguities,
                    "initial_edge_normal_ambiguities": geometry.initial_edge_normal_ambiguities,
                    "description": self.description}

        def reset_cache(self):
            geometry.clear_history()
            self.cache, self.gpu_cache = None, None

        def close(self):
            self.solver._kernel_solve_volume = self.previous_solve

    return SeamContacts()
