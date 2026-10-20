"""Upright base-pinned soft teddy, two Franka arms, and cotton wrapping cloth."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import time
import tomllib
import traceback

import numpy as np
from PIL import Image
import trimesh

from pbd_adapter import existing_pbd_mesh, install_rigid_pbd_friction, simulation_connectivity, install_exact_link_attachments

HERE = Path(__file__).resolve().parent


def write_obj(path, points, faces, uv=None, seam=False):
    if uv is not None:
        texture_uv = uv.copy()
        uv_faces = faces.copy()
        if seam:
            corner_uv = uv[faces]
            extra = {}
            for i in np.flatnonzero(np.ptp(corner_uv[..., 0], axis=1) > .5):
                for corner in np.flatnonzero(corner_uv[i, :, 0] < .5):
                    vertex = int(faces[i, corner])
                    if vertex not in extra:
                        extra[vertex] = len(uv) + len(extra)
                    uv_faces[i, corner] = extra[vertex]
            if extra:
                texture_uv = np.concatenate([uv, uv[list(extra)] + [1, 0]])
    with path.open("w") as f:
        for p in points:
            f.write("v " + " ".join(f"{v:.10g}" for v in p) + "\n")
        if uv is not None:
            for p in texture_uv:
                f.write("vt " + " ".join(f"{v:.10g}" for v in p) + "\n")
        for j, face in enumerate(faces + 1):
            f.write("f " + " ".join(f"{i}/{uv_faces[j,k]+1}" if uv is not None else str(i) for k,i in enumerate(face)) + "\n")


def read_assets(config, run):
    mesh_dir = run / "meshes"
    mesh_dir.mkdir()
    data = {}
    for name in ("bear", "cotton"):
        with np.load(HERE / "assets" / f"{name}.npz", allow_pickle=False) as archive:
            data[name] = {k: archive[k].copy() for k in archive.files}
    bear, cloth = data["bear"], data["cotton"]
    bh = config["table"]["height"] + config["bear"]["base_clearance"]
    bear_offset = np.array([*config["bear"]["position_xy"], bh])
    bear["points"] += bear_offset
    bear["visual_points"] += bear_offset
    write_obj(mesh_dir / "bear.obj", bear["visual_points"], bear["visual_faces"], bear["uv"], seam=True)
    cloth["points"] += [*config["cloth"]["position_xy"], config["table"]["height"] + config["cloth"]["initial_clearance"]]
    write_obj(mesh_dir / "cotton.obj", cloth["points"], cloth["faces"], cloth.get("uv"))
    return data, mesh_dir


def build_scene(config, data, mesh_dir, args, report):
    import genesis as gs
    import torch

    backend = args.backend
    if backend == "auto":
        backend = "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --backend cpu on this machine.")
    print(f"Genesis backend: {backend}", flush=True)
    gs.init(backend=gs.cuda if backend == "cuda" else gs.cpu, seed=config["seed"],
            precision="32", logging_level=logging.WARNING, debug=bool(os.environ.get("BEAR_DEBUG")))
    sim = config["simulation"]
    overview = config["cameras"]["overview"]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim["substep_dt"] * sim["substeps"], substeps=sim["substeps"]),
        rigid_options=gs.options.RigidOptions(enable_self_collision=getattr(args,"task","scene")=="wrap"),
        pbd_options=gs.options.PBDOptions(particle_size=sim["particle_size"],
            hash_grid_res=tuple(sim["hash_grid_size"]),
            max_stretch_solver_iterations=sim["stretch_iterations"],
            max_bending_solver_iterations=sim["bending_iterations"],
            max_volume_solver_iterations=sim["volume_iterations"],
            lower_bound=(-1.2, -1.0, 0), upper_bound=(1.2, 1.0, 2.2)),
        viewer_options=gs.options.ViewerOptions(res=(args.width, args.height),
            camera_pos=overview["position"], camera_lookat=overview["target"], camera_fov=overview["fov"], refresh_rate=30),
        vis_options=gs.options.VisOptions(ambient_light=(.32,.32,.32)),
        show_viewer=not args.headless,
    )
    report["rigid_pbd_contact"] = install_rigid_pbd_friction(scene)
    if getattr(args,"task","scene")=="wrap":
        report["grasp_constraints"]=install_exact_link_attachments(scene)
    rigid_material = gs.materials.Rigid(needs_coup=True, friction=.7, coup_friction=.7)

    def box(name, pos, size, color):
        return scene.add_entity(name=name, morph=gs.morphs.Box(pos=pos, size=size, fixed=True),
            material=rigid_material, surface=gs.surfaces.Default(color=color, roughness=.9))

    scene.add_entity(name="floor", morph=gs.morphs.Plane(), material=rigid_material,
        surface=gs.surfaces.Default(color=(.16, .19, .22, 1)))
    th = config["table"]["height"]
    sx, sy, sz = config["table"]["size"]
    args.table_entity=box("tabletop", (0, 0, th - sz / 2), (sx, sy, sz), (.58, .43, .29, 1))
    for i, x in enumerate((-sx/2+.09, sx/2-.09)):
        for j, y in enumerate((-sy/2+.09, sy/2-.09)):
            box(f"leg_{i}_{j}", (x, y, (th-sz)/2), (.065, .065, th-sz), (.2,.23,.25,1))

    robots = {}
    rc = config["robots"]
    for side in ("left", "right"):
        bp = rc[f"{side}_base"]
        yaw = rc[f"{side}_yaw"]
        box(f"{side}_mount", (bp[0],bp[1],(th+bp[2])/2), (.20,.20,bp[2]-th), (.25,.28,.31,1))
        robots[side] = scene.add_entity(name=f"{side}_franka",
            morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=bp,
                quat=(np.cos(yaw/2),0,0,np.sin(yaw/2))),
            material=gs.materials.Rigid(needs_coup=True, coup_friction=.7, gravity_compensation=1.0))

    bear_data, cloth_data = data["bear"], data["cotton"]
    bcfg, ccfg = config["bear"], config["cloth"]
    tetra = bear_data["tetra"]
    p = bear_data["points"]
    volume = float(np.abs(np.linalg.det(p[tetra[:,1:]]-p[tetra[:,:1]])).sum()/6)
    cloth_area = float(trimesh.Trimesh(vertices=cloth_data["points"], faces=cloth_data["faces"], process=False).area)
    cloth_mass = ccfg["areal_density"] * cloth_area
    bear_material = gs.materials.PBD.Elastic(rho=bcfg["mass"]/volume,
        stretch_compliance=bcfg["stretch_compliance"], volume_compliance=bcfg["volume_compliance"],
        stretch_relaxation=bcfg["stretch_relaxation"], volume_relaxation=bcfg["volume_relaxation"],
        static_friction=bcfg["friction"], kinetic_friction=bcfg["friction"])
    cloth_material = gs.materials.PBD.Cloth(rho=ccfg["areal_density"],
        stretch_compliance=ccfg["stretch_compliance"], bending_compliance=ccfg["bending_compliance"],
        static_friction=ccfg["friction"], kinetic_friction=ccfg["friction"])
    with existing_pbd_mesh("bear", p, tetra, bcfg["mass"]):
        bear = scene.add_entity(name="soft_bear_with_arms", material=bear_material,
            morph=gs.morphs.Mesh(file=str(mesh_dir/"bear.obj"),file_meshes_are_zup=True,decimate=False,convexify=False),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(HERE/"assets"/"bear_fur.png")),
                vis_mode="visual", roughness=1.0))
    with existing_pbd_mesh("cloth", cloth_data["points"], cloth_data["faces"], cloth_mass):
        cloth = scene.add_entity(name="rectangular_cotton", material=cloth_material,
            morph=gs.morphs.Mesh(file=str(mesh_dir/"cotton.obj"),file_meshes_are_zup=True,decimate=False,convexify=False),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(HERE/"assets"/"cotton_weave.png")),
                vis_mode="visual", roughness=1., double_sided=True))
    if getattr(args, "task", "scene") == "wrap":
        from wrap_contact import install_surface_contacts
        args.surface_contacts = install_surface_contacts(scene, bear, cloth,
            bear_data["points"], bear_data["tetra"], cloth_data["points"], cloth_data["faces"], margin=.002,
            iterations=8, max_correction=.001, broadphase=.020, body_response_scale=0.,
            friction=.2)
        from seam_surface_contact import install as install_seam_contacts
        args.seam_contacts = install_seam_contacts(scene, cloth, cloth_data["points"],
            cloth_data["faces"], config.get("seam_surface_contact", {}))
        from velcro_contact import install as install_velcro
        args.velcro = install_velcro(scene, cloth, cloth_data["points"], cloth_data["faces"], config.get("velcro", {}))
        drag = gs.force_fields.Drag(linear=1.0)
        scene.add_force_field(drag)
        drag.activate()
    cameras = {name: scene.add_camera(res=(args.width,args.height), GUI=False,
        pos=cfg["position"], lookat=cfg["target"], fov=cfg["fov"])
        for name,cfg in config["cameras"].items()}
    print("Building scene and compiling physics kernels...", flush=True)
    scene.build()
    fixed = np.flatnonzero(p[:,2] <= p[:,2].min()+bcfg["fixed_band_height"])
    if not 3 <= len(fixed) < .2 * len(p):
        raise RuntimeError(f"Unexpected fixed base particle count: {len(fixed)} / {len(p)}")
    bear.fix_particles(particles_idx_local=fixed, zero_velocity=True)
    initial = {"bear": bear.get_particles_pos().cpu().numpy().copy(), "cloth": cloth.get_particles_pos().cpu().numpy().copy()}
    for name, entity, points, elems, mass in (
        ("bear", bear, p, tetra, bcfg["mass"]),
        ("cloth", cloth, cloth_data["points"], cloth_data["faces"], cloth_mass)):
        assert np.allclose(initial[name], points, rtol=0, atol=2e-7), f"{name} vertex coordinates changed"
        assert np.array_equal(np.sort(simulation_connectivity(entity,name),axis=1), np.sort(elems,axis=1)), f"{name} topology changed"
        assert np.isclose(float(entity.get_mass().cpu().reshape(-1)[0]),mass,rtol=1e-5), f"{name} mass changed"
    home = np.array([*rc["home"], rc["finger_opening"], rc["finger_opening"]],dtype=np.float32)
    for robot in robots.values():
        dofs = [robot.get_joint(name).dofs_idx_local[0] for name in
                [f"joint{i}" for i in range(1,8)]+["finger_joint1","finger_joint2"]]
        robot.set_dofs_position(home, dofs_idx_local=dofs)
        robot.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]),dofs)
        robot.set_dofs_kv(np.array([450,450,350,350,200,200,200,10,10]),dofs)
        limit = np.array([87,87,87,87,12,12,12,100,100])
        robot.set_dofs_force_range(-limit,limit,dofs)
        robot.control_dofs_position(home,dofs)
    report["simulation"] = {"backend":backend,"genesis_version":version("genesis-world"),
        "step_dt_s":sim["substep_dt"]*sim["substeps"],"substep_dt_s":sim["substep_dt"],
        "bear_model":"PBD.Elastic", "cloth_model":"PBD.Cloth", "cotton_calibrated":False,
        "bear_particles":bear.n_particles,"bear_tetrahedra":bear.n_elems,
        "cloth_particles":cloth.n_particles,"cloth_mass_kg":cloth_mass,
        "fixed_particle_count":len(fixed),"free_bear_particles":len(p)-len(fixed),
        "fixed_particle_indices":fixed.tolist(),"fixed_band_height_m":bcfg["fixed_band_height"],
        "robot_dofs":{k:r.n_dofs for k,r in robots.items()},"import_geometry_and_mass_verified":True}
    return scene, robots, bear, cloth, cameras, fixed, initial


def snapshot(bear, cloth, fixed, initial, config):
    out = {}
    for name, entity in (("bear",bear),("cloth",cloth)):
        p=entity.get_particles_pos().cpu().numpy()
        v=entity.get_particles_vel().cpu().numpy()
        if not np.isfinite(p).all() or not np.isfinite(v).all():
            raise RuntimeError(f"Non-finite state: {name}")
        out[name] = {"bounds_m":[p.min(0).tolist(),p.max(0).tolist()],
            "max_displacement_m":float(np.linalg.norm(p-initial[name],axis=1).max()),
            "rms_speed_m_s":float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),
            "table_penetration_m":float(max(0,config["table"]["height"]-p[:,2].min()))}
        if name=="bear":
            out[name]["base_max_displacement_m"] = float(np.linalg.norm(p[fixed]-initial[name][fixed],axis=1).max())
            out[name]["height_ratio"] = float(np.ptp(p[:,2])/np.ptp(initial[name][:,2]))
            head = initial[name][:,2] > initial[name][:,2].min()+.34
            out[name]["head_horizontal_drift_m"] = float(np.linalg.norm((p[head]-initial[name][head]).mean(0)[:2]))
    return out


def render(cameras, run, label):
    for name,camera in cameras.items():
        rgb,*_ = camera.render()
        rgb=np.asarray(rgb)
        if rgb.std()<1:
            raise RuntimeError("Empty rendered image")
        Image.fromarray(rgb).save(run/f"{label}-{name}.png")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task",choices=("scene","wrap"),default="scene")
    ap.add_argument("--backend",choices=("auto","cpu","cuda"),default="auto")
    ap.add_argument("--headless",action="store_true")
    ap.add_argument("--steps",type=int,default=300,help="Settle steps, then keep viewer running")
    ap.add_argument("--width",type=int,default=960)
    ap.add_argument("--height",type=int,default=720)
    ap.add_argument("--config",type=Path,default=HERE/"config.toml")
    ap.add_argument("--output",type=Path)
    ap.add_argument("--probe-deformation",action="store_true",help="After settling apply a local velocity impulse and check soft deformation")
    ap.add_argument("--no-render",action="store_true",help="Skip image output for physics diagnostics")
    ap.add_argument("--record-fps",type=int,default=15)
    ap.add_argument("--stop-after",choices=("grasp","lift","approach_body","wrap","press","thread_arms","release","hold"),default="thread_arms")
    ap.add_argument("--resume",type=Path,help="Resume a saved wrapping stage for development")
    ap.add_argument("--debug-extraction",action="store_true",help="Development only: continue failed press checks to inspect extraction; final result stays failed")
    args=ap.parse_args()
    if args.steps<0 or min(args.width,args.height)<32:
        ap.error("steps must be nonnegative and resolution >=32")
    if args.headless:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cache=Path(os.environ.get("BEAR_CACHE_DIR",HERE/".cache"))
    for key,sub in (("NUMBA_CACHE_DIR","numba"),("GS_CACHE_FILE_PATH","genesis"),("XDG_CACHE_HOME",""),("MPLCONFIGDIR","matplotlib")):
        os.environ.setdefault(key,str(cache/sub))
    if args.task=="wrap":
        from wrap_task import run_wrap
        return run_wrap(args)
    run=args.output or HERE/"runs"/(datetime.now().strftime("%Y%m%d-%H%M%S")+f"-{os.getpid()}")
    run.mkdir(parents=True,exist_ok=False)
    config=tomllib.loads(args.config.read_text())
    report={"status":"starting","config":config,"geometry":json.loads((HERE/"assets"/"metadata.json").read_text()),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/"assets").iterdir() if p.is_file()},
        "scope":"Environment setup; robots hold home pose. No autonomous wrapping task is executed."}
    start=time.monotonic()
    try:
        data,mesh_dir=read_assets(config,run)
        scene,robots,bear,cloth,cameras,fixed,initial=build_scene(config,data,mesh_dir,args,report)
        if not args.no_render:
            render(cameras,run,"initial")
        trace=[]
        max_base_drift=0.
        for i in range(args.steps):
            scene.step()
            base=bear.get_particles_pos().cpu().numpy()[fixed]
            max_base_drift=max(max_base_drift,float(np.linalg.norm(base-initial["bear"][fixed],axis=1).max()))
            if max_base_drift>1e-6:
                raise RuntimeError(f"Fixed base moved by {max_base_drift} m")
            if (i+1)%50==0 or i==args.steps-1:
                state=snapshot(bear,cloth,fixed,initial,config)
                trace.append({"step":i+1,**state})
                print(f"Step {i+1}/{args.steps}: base drift {max_base_drift:.2g} m, height ratio {state['bear']['height_ratio']:.4f}",flush=True)
        state=snapshot(bear,cloth,fixed,initial,config)
        report["settled_state"]=state
        report["trace"]=trace
        report["max_base_displacement_over_run_m"]=max_base_drift
        report["simulated_seconds"]=args.steps*report["simulation"]["step_dt_s"]
        if not args.no_render:
            render(cameras,run,"settled")
        np.savez_compressed(run/"settled_state.npz", bear_positions=bear.get_particles_pos().cpu().numpy(),
            bear_velocities=bear.get_particles_vel().cpu().numpy(),cloth_positions=cloth.get_particles_pos().cpu().numpy(),
            fixed_indices=fixed,**{k+"_qpos":r.get_qpos().cpu().numpy() for k,r in robots.items()})
        robot_states={k:r.get_qpos().cpu().numpy() for k,r in robots.items()}
        robot_finite=all(np.isfinite(q).all() for q in robot_states.values())
        checks={"fixed_base_immobile":max_base_drift<=1e-6,
            "bear_stays_upright":.85<state["bear"]["height_ratio"]<1.15 and state["bear"]["head_horizontal_drift_m"]<.05,
            "no_table_penetration":max(v["table_penetration_m"] for v in state.values())<.001,
            "two_franka_arms":len(robots)==2,"finite_states":robot_finite}
        if args.probe_deformation:
            rest=bear.get_particles_pos().cpu().numpy().copy()
            local_z=rest[:,2]-rest[:,2].min()
            probe=np.flatnonzero((local_z>.18)&(local_z<.32)&(rest[:,0]>.03))
            if len(probe)==0:
                raise RuntimeError("No particles selected for deformation probe")
            velocity=np.tile([.22,0.,0.],(len(probe),1)).astype(np.float32)
            bear.set_particles_vel(velocity,particles_idx_local=probe)
            max_motion=0.
            max_strain=0.
            max_probe_base=0.
            edges=np.asarray(bear._edges)
            rest_lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
            if np.any(rest_lengths<=1e-8):
                raise RuntimeError("Degenerate edge in deformation probe")
            for _ in range(100):
                scene.step()
                current=bear.get_particles_pos().cpu().numpy()
                if not np.isfinite(current).all():
                    raise RuntimeError("Probe produced non-finite state")
                max_motion=max(max_motion,float(np.linalg.norm(current-rest,axis=1).max()))
                max_probe_base=max(max_probe_base,float(np.linalg.norm(current[fixed]-initial["bear"][fixed],axis=1).max()))
                lengths=np.linalg.norm(current[edges[:,0]]-current[edges[:,1]],axis=1)
                max_strain=max(max_strain,float(np.abs(lengths/rest_lengths-1).max()))
            report["deformation_probe"]={"type":"single local velocity impulse, then free simulation",
                "impulse_speed_m_s":.22,"affected_particles":len(probe),"max_particle_displacement_m":max_motion,
                "max_relative_edge_length_change":max_strain,"base_max_displacement_m":max_probe_base}
            checks["soft_body_responds_to_impulse"]=max_motion>1e-4 and max_strain>1e-4
            checks["probe_base_immobile"]=max_probe_base<=1e-6
        report["checks"]=checks
        report["status"]="passed" if all(checks.values()) else "failed"
        report["wall_seconds"]=time.monotonic()-start
        (run/"report.json").write_text(json.dumps(report,indent=2))
        if not all(checks.values()):
            raise RuntimeError(f"Scene checks failed: {checks}")
        print(f"PASS: {run}",flush=True)
        if not args.headless:
            print("Viewer active. Close the window or press Ctrl+C to exit.",flush=True)
            while scene.viewer.is_alive():
                scene.step()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        report.update(status="failed",error=str(exc),traceback=traceback.format_exc())
        (run/"report.json").write_text(json.dumps(report,indent=2))
        raise
    finally:
        import genesis as gs
        if gs._initialized:
            gs.destroy()


if __name__=="__main__":
    main()
