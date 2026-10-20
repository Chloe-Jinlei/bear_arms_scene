"""Contact-gated, local hook/loop adhesion for Genesis 1.4.3 PBD cloth.

This is an ideal no-slip adhesive approximation with a geometric detachment
threshold, not a calibrated hook-and-loop force/peel model. A bond stores a
cloth vertex relative to a triangle on the complementary strip, including its
actual activation gap. There is no world anchor, pose target, or long-range
attraction. Only free participating nodes receive inverse-mass corrections.
"""
from importlib.metadata import version
import hashlib
import json

import numpy as np
import trimesh


DEFAULTS = {
    "strip_width_m": .04,
    "strip_inset_m": 0.,
    "surface_signs": {"left": 1, "right": -1},
    "activation_distance_m": .005,
    "normal_angle_tolerance_deg": 35.,
    "facing_tolerance_m": .001,
    "dwell_time_s": .08,
    "break_error_m": .015,
    "iterations_per_substep": 3,
    "max_correction_per_pass_m": .0003,
    "relaxation": .8,
    "break_dwell_time_s": .08,
}


def face_frames(triangles):
    tangent = triangles[:, 1] - triangles[:, 0]
    normal = np.cross(tangent, triangles[:, 2] - triangles[:, 0])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return np.stack([tangent, np.cross(normal, tangent), normal], axis=2)


class VelcroGeometry:
    """Pure geometry shared by the contact gate and independent inspection."""

    def __init__(self, points, faces, config=None):
        supplied = config or {}
        supplied = supplied.get("velcro", supplied)
        self.config = {**DEFAULTS, **supplied}
        self.config["surface_signs"] = {**DEFAULTS["surface_signs"], **supplied.get("surface_signs", {})}
        if any(value not in (-1, 1) for value in self.config["surface_signs"].values()):
            raise ValueError("Velcro surface signs must each be +1 or -1")
        if not (0 < self.config["activation_distance_m"] < self.config["break_error_m"]
                and self.config["dwell_time_s"] > 0
                and 0 < self.config["relaxation"] <= 1):
            raise ValueError("Invalid velcro distance, dwell, detachment, or relaxation parameters")
        self.rest = np.asarray(points, dtype=float).copy()
        self.faces = np.asarray(faces, dtype=np.int32).copy()
        width = float(self.config["strip_width_m"])
        inset = float(self.config["strip_inset_m"])
        if not (np.isfinite(inset) and inset >= 0
                and np.isfinite(width) and width > 0
                and inset + width < np.ptp(self.rest[:, 0]) / 2):
            raise ValueError("Velcro inset must be nonnegative and inset plus width less than half the cloth length")
        left_depth = self.rest[:, 0] - self.rest[:, 0].min()
        right_depth = self.rest[:, 0].max() - self.rest[:, 0]
        self.strips = {
            "left": np.flatnonzero((left_depth >= inset - 1e-9) & (left_depth <= inset + width + 1e-9)),
            "right": np.flatnonzero((right_depth >= inset - 1e-9) & (right_depth <= inset + width + 1e-9)),
        }
        self.strip_faces = {side: np.flatnonzero(np.isin(self.faces, nodes).all(1))
                            for side, nodes in self.strips.items()}
        if any(not len(f) for f in self.strip_faces.values()):
            raise ValueError("Velcro strip contains no complete triangles at this mesh resolution")
        self.source_nodes = self.strips["left"]
        self.target_face_ids = self.strip_faces["right"]
        self.target_faces = self.faces[self.target_face_ids]
        incident=[self.strip_faces['left'][np.any(self.faces[self.strip_faces['left']]==v,axis=1)] for v in self.source_nodes]
        width=max(map(len,incident));self.incident_faces=np.zeros((len(incident),width),dtype=int);self.incident_valid=np.zeros((len(incident),width),bool)
        for k,ids in enumerate(incident):self.incident_faces[k,:len(ids)]=ids;self.incident_valid[k,:len(ids)]=True
        self.area_weight = np.zeros(len(points))
        patch_tri = self.rest[self.faces[self.strip_faces["left"]]]
        patch_area = np.linalg.norm(np.cross(patch_tri[:, 1] - patch_tri[:, 0],
                                            patch_tri[:, 2] - patch_tri[:, 0]), axis=1) / 2
        np.add.at(self.area_weight, self.faces[self.strip_faces["left"]].ravel(), np.repeat(patch_area / 3, 3))
        self.topology_sha256 = hashlib.sha256(self.rest.tobytes() + self.faces.tobytes()).hexdigest()

    def candidates(self, points):
        """Measure distance, facing and alignment without moving any particle."""
        points = np.asarray(points, dtype=float)
        full = trimesh.Trimesh(points, self.faces, process=False)
        # A folded strip can place several triangles near one hook vertex.
        # The nearest triangle may face away while another is in valid contact.
        # Enumerate this small strip patch instead of discarding such contacts.
        source=points[self.source_nodes]
        target_triangles=points[self.target_faces]
        incident_normals=face_frames(points[self.faces])[:,:,2][self.incident_faces]*self.config['surface_signs']['left']
        nb=face_frames(target_triangles)[:,:,2]*self.config['surface_signs']['right']
        count_a,count_b=len(source),len(target_triangles)
        all_closest=trimesh.triangles.closest_point(
            np.tile(target_triangles,(count_a,1,1)),np.repeat(source,count_b,axis=0)).reshape(count_a,count_b,3)
        diff=source[:,None]-all_closest
        all_distance=np.linalg.norm(diff,axis=2)
        dots=np.einsum('aki,bi->abk',incident_normals,nb)
        gaps=np.einsum('aki,abi->abk',incident_normals,-diff)
        gb=np.einsum('bi,abi->ab',nb,diff)
        source_valid=self.incident_valid[:,None,:]
        source_facing=source_valid&(dots<=-np.cos(np.deg2rad(self.config['normal_angle_tolerance_deg'])))&(gaps>=-self.config['facing_tolerance_m'])
        choices=np.argmin(np.where(np.where(source_facing.any(2)[...,None],source_facing,source_valid),dots,np.inf),axis=2)
        all_dot=np.take_along_axis(dots,choices[...,None],axis=2)[...,0]
        ga=np.take_along_axis(gaps,choices[...,None],axis=2)[...,0]
        facing=(all_dot<=-np.cos(np.deg2rad(self.config['normal_angle_tolerance_deg'])))
        facing&=np.minimum(ga,gb)>=-self.config['facing_tolerance_m']
        valid=facing&(all_distance<=self.config['activation_distance_m'])
        nearby_facing=facing&(all_distance<.08)
        feedback=(all_dot<0)&(np.minimum(ga,gb)>=-self.config['facing_tolerance_m'])&(all_distance<.08)
        preferred=np.where(valid.any(1)[:,None],valid,
                    np.where(feedback.any(1)[:,None],feedback,np.ones_like(valid)))
        angle_error=np.maximum(0,np.arccos(np.clip(-all_dot,-1,1))-np.deg2rad(self.config['normal_angle_tolerance_deg']))
        cost=all_distance+np.where(valid.any(1)[:,None],0,.03*angle_error)
        local_face=np.argmin(np.where(preferred,cost,np.inf),axis=1)
        closest=all_closest[np.arange(count_a),local_face]
        distance=all_distance[np.arange(count_a),local_face]
        tri = points[self.target_faces[local_face]]
        bary = trimesh.triangles.points_to_barycentric(tri, closest)
        bary = np.clip(bary, 0., 1.)
        bary /= bary.sum(1, keepdims=True)
        frames = face_frames(tri)
        incident_choice=choices[np.arange(count_a),local_face]
        normal_a=incident_normals[np.arange(count_a),incident_choice]
        source_face_ids=self.incident_faces[np.arange(count_a),incident_choice]
        normal_b = frames[:, :, 2] * self.config["surface_signs"]["right"]
        difference = points[self.source_nodes] - closest
        alignment = np.einsum("ij,ij->i", normal_a, normal_b)
        facing_a = np.einsum("ij,ij->i", normal_a, -difference)
        facing_b = np.einsum("ij,ij->i", normal_b, difference)
        accepted = (distance <= self.config["activation_distance_m"])
        accepted &= alignment <= -np.cos(np.deg2rad(self.config["normal_angle_tolerance_deg"]))
        accepted &= np.minimum(facing_a, facing_b) >= -self.config["facing_tolerance_m"]
        return {"source_nodes": self.source_nodes, "source_face_ids":source_face_ids, "target_nodes": self.target_faces[local_face],
                "target_face_ids": self.target_face_ids[local_face], "barycentric": bary,
                "local_offsets": np.einsum("nji,nj->ni", frames, difference),
                "distance_m": distance, "normal_dot": alignment,
                "source_active_normals": normal_a, "target_active_normals": normal_b,
                "source_facing_gap_m": facing_a, "target_facing_gap_m": facing_b,
                "accepted": accepted}

    def errors(self, positions, bonds):
        if not bonds:
            return np.empty(0)
        sources = np.array([b["source_node"] for b in bonds])
        targets = np.array([b["target_nodes"] for b in bonds])
        bary = np.array([b["barycentric"] for b in bonds])
        offsets = np.array([b["local_offset_m"] for b in bonds])
        tri = np.asarray(positions)[targets]
        target_points = (tri * bary[..., None]).sum(1)
        target_points += np.einsum("nij,nj->ni", face_frames(tri), offsets)
        return np.linalg.norm(np.asarray(positions)[sources] - target_points, axis=1)


def install(scene, cloth, points, faces, config=None):
    """Install after surface contacts and before build; close in reverse order.

    Each step: ``velcro.refresh(time_s=t, dt=dt)`` before ``scene.step()``.
    Checkpoints: ``dump()`` and ``restore(saved_dict)`` after state restoration.
    """
    import genesis as gs
    import quadrants as qd
    import torch

    if version("genesis-world") != "1.4.3" or scene.is_built:
        raise RuntimeError("Install the Genesis 1.4.3 velcro adapter before scene.build()")
    geometry = VelcroGeometry(points, faces, config)

    @qd.data_oriented
    class VelcroContacts:
        def __init__(self):
            self.geometry, self.config = geometry, geometry.config
            self.solver, self.cloth = cloth.solver, cloth
            self.previous_solve = self.solver._kernel_solve_volume
            self.solver._kernel_solve_volume = self.solve
            self.bonds, self.pending = [], {}
            self.cache = None
            self.cache_dirty = False
            self.time_s = 0.
            self.last_refresh_time = None
            self.latest_candidate_count = 0
            self.latest_min_strip_distance = None
            self.enabled = True
            edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
            end_nodes=np.abs(points[:,0]-np.mean([points[:,0].min(),points[:,0].max()]))>np.ptp(points[:,0])/2-.10
            edges=edges[np.any(end_nodes[edges],axis=1)]
            self.local_edges=torch.as_tensor(np.ascontiguousarray(edges+cloth.particle_start),device=gs.device,dtype=gs.tc_int)
            self.local_lengths=torch.as_tensor(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1),device=gs.device,dtype=gs.tc_float)
            self.description = {
                "model": "contact-gated ideal local no-slip hook/loop adhesion with geometric detachment",
                "calibrated_force_or_peel_model": False,
                "coupled_local_stretch_iterations":"Alternate adhesion with local native-rest-length cloth constraints within 10 cm of the joined ends to reduce solver stress concentration; no additional pins or bonds.",
                "strip_width_m": self.config["strip_width_m"],
                "strip_inset_m": self.config["strip_inset_m"],
                "strip_nodes": {s: a.tolist() for s, a in geometry.strips.items()},
                "surface_signs": self.config["surface_signs"],
                "activation_distance_m": self.config["activation_distance_m"],
                "dwell_time_s": self.config["dwell_time_s"],
                "normal_angle_tolerance_deg": self.config["normal_angle_tolerance_deg"],
                "break_error_m": self.config["break_error_m"],
                "break_dwell_time_s":self.config['break_dwell_time_s'],
                "detachment_measurement":"Committed control-step states after all constraint passes, not intermediate solver residuals",
                "local_activation_gap_preserved": True,
                "source_contact_normal":"Actual incident hook-face normal, not a smoothed vertex normal across a fold; target uses its actual loop-face normal.",
                "reciprocal_inverse_mass_corrections": True,
                "world_anchors": False, "long_range_attraction": False,
                "bonded_source_vertex_area_m2": float(geometry.area_weight.sum()),
                "maximum_bonds": len(geometry.source_nodes),
            }

        def _positions(self):
            return cloth.get_particles_pos().detach().cpu().numpy()

        def _sync_status(self):
            if self.cache is not None:
                active = self.cache[4].detach().cpu().numpy()
                for bond, flag in zip(self.bonds, active):
                    if bond["active"] and not flag:
                        bond["active"] = False
                        self.cache_dirty = True
                        bond["detected_broken_at_time_s"] = self.time_s
                        bond["break_reason"] = "relative vertex/triangle error exceeded geometric detachment threshold"

        def _rebuild_cache(self):
            if not self.bonds:
                self.cache = None
                return
            start = cloth.particle_start
            src = np.array([b["source_node"] + start for b in self.bonds], dtype=np.int32)
            dst = np.array([b["target_nodes"] for b in self.bonds], dtype=np.int32) + start
            bary = np.array([b["barycentric"] for b in self.bonds])
            offset = np.array([b["local_offset_m"] for b in self.bonds])
            active = np.array([b["active"] for b in self.bonds], dtype=np.int32)
            counts = np.bincount(np.concatenate([src[active > 0], dst[(bary > 1e-10) & (active[:, None] > 0)]]),
                                 minlength=self.solver.n_particles)
            values = [src, dst, bary, offset, active, counts]
            self.cache = [torch.as_tensor(np.ascontiguousarray(value), device=gs.device,
                          dtype=gs.tc_int if i in (0, 1, 4, 5) else gs.tc_float)
                          for i, value in enumerate(values)]
            self.cache_dirty = False

        def refresh(self, time_s=None, dt=None, positions=None):
            dt = float(scene.sim.dt if dt is None else dt)
            self.time_s = self.time_s + dt if time_s is None else float(time_s)
            if self.last_refresh_time is not None and self.time_s < self.last_refresh_time - 1e-9:
                raise ValueError("Velcro contact time must be nondecreasing")
            if self.last_refresh_time is not None and self.time_s - self.last_refresh_time > 1.5 * dt + 1e-9:
                # Unobserved time is not evidence of sustained surface contact.
                self.pending.clear()
            self._sync_status()
            positions = self._positions() if positions is None else np.asarray(positions)
            active_bonds=[b for b in self.bonds if b['active']]
            for bond,error in zip(active_bonds,geometry.errors(positions,active_bonds)):
                if error>self.config['break_error_m']:
                    bond.setdefault('over_error_since_s',self.time_s)
                    if self.time_s-bond['over_error_since_s']>=self.config['break_dwell_time_s']-1e-9:
                        bond['active']=False;self.cache_dirty=True
                        bond['detected_broken_at_time_s']=self.time_s
                        bond['break_reason']='sustained geometric separation in committed states'
                else:bond.pop('over_error_since_s',None)
            candidates = geometry.candidates(positions)
            self.latest_candidate_count = int(candidates["accepted"].sum())
            self.latest_min_strip_distance = float(candidates["distance_m"].min())
            used = {b["source_node"] for b in self.bonds}
            changed = False
            # Broken bonds do not automatically reattach. A fresh physical
            # contact cycle must be explicitly introduced by a later experiment.
            for k, source in enumerate(candidates["source_nodes"]):
                source = int(source)
                accepted = self.enabled and bool(candidates["accepted"][k]) and source not in used
                if not accepted:
                    self.pending.pop(source, None)
                    continue
                if source not in self.pending:
                    self.pending[source] = self.time_s
                elapsed = self.time_s - self.pending[source]
                if elapsed + 1e-9 < self.config["dwell_time_s"]:
                    continue
                self.bonds.append({"id": len(self.bonds), "source_node": source,
                    "target_nodes": candidates["target_nodes"][k].tolist(),
                    "target_face_id": int(candidates["target_face_ids"][k]),
                    "barycentric": candidates["barycentric"][k].tolist(),
                    "local_offset_m": candidates["local_offsets"][k].tolist(),
                    "active": True, "activation_time_s": self.time_s,
                    "continuous_contact_start_s": self.pending.pop(source), "dwell_s": elapsed,
                    "activation_distance_m": float(candidates["distance_m"][k]),
                    "activation_normal_dot": float(candidates["normal_dot"][k]),
                    "source_active_face_id":int(candidates['source_face_ids'][k]),
                    "source_active_normal": candidates["source_active_normals"][k].tolist(),
                    "target_active_normal": candidates["target_active_normals"][k].tolist(),
                    "source_facing_gap_m": float(candidates["source_facing_gap_m"][k]),
                    "target_facing_gap_m": float(candidates["target_facing_gap_m"][k])})
                changed = True
            if changed or self.cache_dirty:
                self._rebuild_cache()
            self.last_refresh_time = self.time_s
            return self.metrics(positions=positions)

        def solve(self, f):
            # The saved hook includes moving soft-bear surface contact. Preserve
            # it exactly and append adhesion in the same physical substep.
            self.previous_solve(f)
            if self.cache is not None:
                for _ in range(int(self.config["iterations_per_substep"])):
                    self.project(*self.cache)
                    for _ in range(4):self.project_local_stretch(self.local_edges,self.local_lengths)
                self.project(*self.cache)

        @qd.kernel
        def project_local_stretch(self,edges:qd.types.ndarray(),lengths:qd.types.ndarray()):
            for k in range(edges.shape[0]):
                a,b=edges[k,0],edges[k,1]
                wa=self.solver.particles[a,0].free/self.solver.particles_info[a].mass
                wb=self.solver.particles[b,0].free/self.solver.particles_info[b].mass
                delta=self.solver.particles[a,0].pos-self.solver.particles[b,0].pos
                distance=delta.norm()
                correction=-.2*(distance-lengths[k])/(wa+wb+.1)*delta/qd.max(distance,1e-12)
                self.solver.particles[a,0].dpos+=wa*correction
                self.solver.particles[b,0].dpos-=wb*correction
            for i in range(cloth.particle_start,cloth.particle_end):
                if self.solver.particles[i,0].free:self.solver.particles[i,0].pos+=self.solver.particles[i,0].dpos
                self.solver.particles[i,0].dpos.fill(0.)

        @qd.kernel
        def project(self, src: qd.types.ndarray(), dst: qd.types.ndarray(),
                    bary: qd.types.ndarray(), offsets: qd.types.ndarray(),
                    active: qd.types.ndarray(), counts: qd.types.ndarray()):
            for k in range(src.shape[0]):
                if active[k]:
                    si = src[k]
                    a = self.solver.particles[dst[k, 0], 0].pos
                    b = self.solver.particles[dst[k, 1], 0].pos
                    c = self.solver.particles[dst[k, 2], 0].pos
                    tangent = (b - a).normalized(1e-12)
                    normal = (b - a).cross(c - a).normalized(1e-12)
                    offset = tangent * offsets[k, 0] + normal.cross(tangent) * offsets[k, 1] + normal * offsets[k, 2]
                    target = offset
                    sw = self.solver.particles[si, 0].free / self.solver.particles_info[si].mass
                    denominator = sw
                    multiplier_limit = gs.qd_float(1e20)
                    if sw > 0:
                        multiplier_limit = self.config["max_correction_per_pass_m"] / (sw * qd.max(counts[si], 1))
                    for j in qd.static(range(3)):
                        di = dst[k, j]
                        dw = self.solver.particles[di, 0].free / self.solver.particles_info[di].mass
                        target += bary[k, j] * self.solver.particles[di, 0].pos
                        denominator += dw * bary[k, j] ** 2
                        if dw * bary[k, j] > 1e-12:
                            multiplier_limit = qd.min(multiplier_limit, self.config["max_correction_per_pass_m"] / (dw * bary[k, j] * qd.max(counts[di], 1)))
                    error = self.solver.particles[si, 0].pos - target
                    length = error.norm()
                    if length > 1e-10 and denominator > 1e-12:
                        magnitude = qd.min(self.config["relaxation"] * length / denominator, multiplier_limit)
                        impulse = magnitude * error / length
                        self.solver.particles[si, 0].dpos -= sw * impulse
                        for j in qd.static(range(3)):
                            di = dst[k, j]
                            dw = self.solver.particles[di, 0].free / self.solver.particles_info[di].mass
                            self.solver.particles[di, 0].dpos += dw * bary[k, j] * impulse
            for i in range(self.solver.n_particles):
                if self.solver.particles[i, 0].free:
                    self.solver.particles[i, 0].pos += self.solver.particles[i, 0].dpos
                self.solver.particles[i, 0].dpos.fill(0.)

        def metrics(self, positions=None):
            self._sync_status()
            positions = self._positions() if positions is None else positions
            current = [b for b in self.bonds if b["active"]]
            errors = geometry.errors(positions, current)
            sources = np.array([b["source_node"] for b in current], dtype=int)
            return {"active_bonds": len(current), "total_activated": len(self.bonds),
                    "broken_bonds": len(self.bonds) - len(current),
                    "max_bond_error_m": float(errors.max()) if len(errors) else 0.,
                    "mean_bond_error_m": float(errors.mean()) if len(errors) else 0.,
                    "latest_contact_candidates": self.latest_candidate_count,
                    "minimum_strip_vertex_to_face_distance_m": self.latest_min_strip_distance,
                    "bonded_hook_vertex_area_m2": float(geometry.area_weight[sources].sum()),
                    "bonded_width_span_m": float(np.ptp(geometry.rest[sources, 1])) if len(sources) else 0.,
                    "activation_events": json.loads(json.dumps(self.bonds)),
                    "model": self.description["model"], "force_or_peel_strength_calibrated": False}

        def dump(self):
            self._sync_status()
            return {"schema": 1, "topology_sha256": geometry.topology_sha256,
                    "config": self.config, "time_s": self.time_s, "enabled": self.enabled,
                    "pending": {str(k): v for k, v in self.pending.items()},
                    "bonds": json.loads(json.dumps(self.bonds))}

        def restore(self, saved):
            if saved.get("schema") != 1 or saved.get("topology_sha256") != geometry.topology_sha256:
                raise ValueError("Velcro checkpoint geometry does not match this cloth")
            saved_config = dict(saved.get("config", {}))
            # Checkpoints made before the optional inset used strips at the ends.
            saved_config.setdefault("strip_inset_m", 0.)
            if saved_config != self.config:
                raise ValueError("Velcro checkpoint contact parameters differ")
            self.bonds = json.loads(json.dumps(saved["bonds"]))
            self.pending = {int(k): float(v) for k, v in saved.get("pending", {}).items()}
            self.time_s = float(saved.get("time_s", 0.))
            self.last_refresh_time = None
            self.enabled = bool(saved.get("enabled", True))
            self._rebuild_cache()

        def close(self):
            self.solver._kernel_solve_volume = self.previous_solve

    return VelcroContacts()


install_velcro_contacts = install
