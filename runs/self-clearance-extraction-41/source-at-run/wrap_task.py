"""Scripted contact-gated dual-Panda cloth wrapping experiment."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp
import trimesh

HERE=Path(__file__).resolve().parent
TCP=np.array([0.,0.,.1029])
PHASES={"settle":"初始静置","pregrasp":"双臂接近布料前侧两角",
    "grasp":"闭爪并抓住两个前角","lift":"提起棉布","approach_body":"靠近熊的腹部",
    "wrap":"沿身体两侧环绕","press":"布边压合 · 魔术贴接触粘接",
    "release":"松开夹爪并撤离","hold":"松爪后检验粘接保持"}

def array(t):
    return t.detach().cpu().numpy().copy()

def matrix(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()

def quat_for_tip(tip, tilt=0.):
    n=np.asarray([tip[0],tip[1]-.08,0.],float)
    n/=max(np.linalg.norm(n),1e-8)
    axis=np.cross([0,0,-1],-n)
    axis/=max(np.linalg.norm(axis),1e-8)
    return (Rotation.from_rotvec(axis*tilt)*Rotation.from_quat([1,0,0,0])).as_quat()[[3,0,1,2]]

class WrapExperiment:
    def __init__(self,scene,robots,bear,cloth,cameras,fixed,initial,data,config,report,run,args):
        self.scene,self.robots,self.bear,self.cloth=scene,robots,bear,cloth
        self.cameras,self.fixed,self.initial,self.data=cameras,fixed,initial,data
        self.config,self.report,self.run,self.args=config,report,run,args
        self.contact=args.surface_contacts
        self.seam_contact=args.seam_contacts
        self.velcro=args.velcro
        self.plan=json.loads((HERE/"wrap_plan.json").read_text())
        self.velcro_plan=json.loads((HERE/"velcro_plan.json").read_text())
        self.velcro_plan.setdefault("samples",[])
        self.velcro_plan.setdefault("waypoints",[])
        self.dt=report["simulation"]["step_dt_s"]
        self.table=config["table"]["height"]
        self.hands={s:r.get_link("hand") for s,r in robots.items()}
        self.fingers={s:[r.get_link(n) for n in ("left_finger","right_finger")] for s,r in robots.items()}
        self.grips={s:np.array([.04,.04]) for s in robots}
        self.quats={s:array(h.get_quat()) for s,h in self.hands.items()}
        self.bindings={}
        self.step_count=0
        self.phase="settle"
        self.max_base_drift=0.
        self.max_arm_contacts=0
        self.max_self_contacts=0
        self.max_table_contacts=0
        self.max_cloth_stretch=1.
        self.max_grasp_error=0.
        self.last_bond_count=0
        self.release_min_bonds=None
        self.next_frame=0.
        self.video=None
        self.left_video=None
        self.rear_video=None
        self.recording_times=[]
        self.replay_frames=[]
        self.log=(run/"trajectory.jsonl").open("w")
        self.font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",20)
        self.small_font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",15)
        self.cloth_edges=np.asarray(cloth.edges)
        cp=data["cotton"]["points"]
        self.rest_lengths=np.linalg.norm(cp[self.cloth_edges[:,0]]-cp[self.cloth_edges[:,1]],axis=1)
        front=np.flatnonzero(np.isclose(cp[:,1],cp[:,1].min()))
        self.front_edge=front[np.argsort(cp[front,0])]
        self.corner={"left":int(self.front_edge[0]),"right":int(self.front_edge[-1])}
        inset=config.get("grasp",{}).get("inset_from_short_edge_m",0.)
        self.grasp_inset={"left":np.array([inset,0.,0.]),"right":np.array([-inset,0.,0.])}
        self.eligible={s:np.flatnonzero(np.linalg.norm(cp-cp[i],axis=1)<.019) for s,i in self.corner.items()}
        self.task=report["wrapping"]={"controller":"Contact-adaptive grasp IK; prevalidated continuous joint trajectory for wrapping",
            "grasp_model":"contact-gated local no-slip PBD link attachments",
            "grasp_friction_or_force_validated":False,
            "object_position_setters_during_action":False,"robot_position_setters_during_action":False,
            "cloth_corners":"two front (negative Y) corners of the initial flat rectangle",
            "corner_particle_indices":self.corner,"phases":[],"grasps":{},
            "grasp_inset_from_short_edge_m":inset,
            "surface_contact":self.contact.description,
            "seam_surface_contact":self.seam_contact.description,
            "completion_scope":("Grasp, lift, wrap and join complementary hook/loop strips; grippers remain holding at the requested press endpoint. Release is not verified."
                if args.stop_after=="press" else "Wrap, make contact between complementary hook/loop strips, release both grippers, and verify the joined cloth stays around the belly.")}
        self.task["velcro"]=self.velcro.metrics()
        if not args.no_render:
            from genesis.utils.video_encoder import VideoEncoder
            self.video=VideoEncoder(str(run/"wrapping-process.mp4"),args.record_fps,codec="libx264",bitrate=4.)
            self.left_video=VideoEncoder(str(run/"wrapping-left-side.mp4"),args.record_fps,codec="libx264",bitrate=4.)
            self.rear_video=VideoEncoder(str(run/"wrapping-rear.mp4"),args.record_fps,codec="libx264",bitrate=4.)
        self.report["recordings"]={"enabled":not args.no_render,"fps":args.record_fps,
            "files":(["wrapping-process.mp4","wrapping-left-side.mp4","wrapping-rear.mp4"] if not args.no_render else []),
            "left_camera":"table_left","left_camera_config":config["cameras"]["table_left"],
            "left_video_resolution":[args.width,args.height],
            "rear_camera":"bear_rear","rear_camera_config":config["cameras"]["bear_rear"],
            "same_simulation_steps_for_all_videos":True,"simulation_times_s":self.recording_times,
            "per_frame_render_state_archive":"replay-frames.npz" if not args.no_render else None}
        for r in robots.values():
            r.set_dofs_kp([500,500],dofs_idx_local=[7,8])
            r.set_dofs_kv([20,20],dofs_idx_local=[7,8])

    def persist(self):
        self.task["completed_steps"]=self.step_count
        self.task["simulated_seconds"]=self.step_count*self.dt
        self.task["velcro"]=self.velcro.metrics()
        self.task["seam_contact_metrics"]=self.seam_contact.metrics()
        (self.run/"report.json").write_text(json.dumps(self.report,indent=2))

    def start(self,name):
        self.phase=name
        self.task["phases"].append({"name":name,"start_seconds":self.step_count*self.dt})
        print(f"Phase {name}: t={self.step_count*self.dt:.2f}s",flush=True)
        self.persist()

    def tcp(self,side):
        h=self.hands[side]
        return array(h.get_pos())+matrix(array(h.get_quat()))@TCP

    def solve(self,side,pos,quat=None):
        robot=self.robots[side]
        quat=self.quats[side] if quat is None else quat
        q,e=robot.inverse_kinematics(link=self.hands[side],pos=pos,quat=quat,
            local_point=TCP,init_qpos=robot.get_qpos(),max_samples=12,max_solver_iters=80,
            pos_tol=5e-4,rot_tol=5e-3,return_error=True)
        e=array(e)
        if np.linalg.norm(e[:3])>.002 or np.linalg.norm(e[3:])>.03:
            q,e=robot.inverse_kinematics(link=self.hands[side],pos=pos,quat=quat,
                local_point=TCP,init_qpos=q,max_samples=40,max_solver_iters=200,
                pos_tol=2e-4,rot_tol=2e-3,return_error=True)
            e=array(e)
        if np.linalg.norm(e[:3])>.003 or np.linalg.norm(e[3:])>.04:
            raise RuntimeError(f"{side} IK failed: target {pos}, error {e}")
        q=array(q);q[7:]=self.grips[side]
        return q

    def frame(self):
        views=[]
        closeup=self.phase in ("press","release","hold")
        extracting=self.phase.startswith(("regrasp","pull")) or self.phase=="thread_arms"
        for name in ("overview","front" if extracting else "back" if closeup else "overhead"):
            rgb,*_=self.cameras[name].render()
            views.append(Image.fromarray(np.asarray(rgb)))
        w,h=views[0].size
        canvas=Image.new("RGB",(2*w,h+76),"#17212b")
        canvas.paste(views[0],(0,40));canvas.paste(views[1],(w,40))
        draw=ImageDraw.Draw(canvas)
        draw.text((14,8),f"软体熊穿孔包裹  |  {PHASES[self.phase]}  |  {self.step_count*self.dt:.2f} s",font=self.font,fill="white")
        metrics=self.velcro.metrics()
        status=f"已粘接 {metrics['active_bonds']} 个局部连接" if metrics['active_bonds'] else "魔术贴等待实际接触"
        view_label="总览 / 正面穿孔视角" if extracting else "总览 / 背面粘接视角" if closeup else "总览 / 俯视"
        draw.text((14,h+46),f"{view_label}    熊底部固定 · {status}    仿真原速",font=self.small_font,fill="#d9e4e9")
        if self.video:self.video.write(np.asarray(canvas))
        if self.left_video:
            left_rgb,*_=self.cameras["table_left"].render()
            self.left_video.write(np.asarray(left_rgb))
        if self.rear_video:
            rear_rgb,*_=self.cameras["bear_rear"].render()
            self.rear_video.write(np.asarray(rear_rgb))
        self.recording_times.append(self.step_count*self.dt)
        self.replay_frames.append({'time_s':self.step_count*self.dt,'phase':self.phase,
            'bear_positions':array(self.bear.get_particles_pos()),'cloth_positions':array(self.cloth.get_particles_pos()),
            'left_qpos':array(self.robots['left'].get_qpos()),'right_qpos':array(self.robots['right'].get_qpos()),
            'active_bonds':metrics['active_bonds']})
        return canvas

    def record(self,bp,cp):
        stretch=np.linalg.norm(cp[self.cloth_edges[:,0]]-cp[self.cloth_edges[:,1]],axis=1)/self.rest_lengths
        entry={"step":self.step_count,"seconds":self.step_count*self.dt,"phase":self.phase,
            "base_max_displacement_m":self.max_base_drift,"cloth_max_stretch_ratio":float(stretch.max()),
            "cloth_bounds_m":[cp.min(0).tolist(),cp.max(0).tolist()],
            "tcp_m":{s:self.tcp(s).tolist() for s in self.robots},
            "qpos":{s:array(r.get_qpos()).tolist() for s,r in self.robots.items()},
            "contact_constraint_count":self.contact.count,
            "seam_contact":self.seam_contact.metrics(positions=cp),
            "velcro_active_bonds":self.velcro.metrics()["active_bonds"]}
        self.log.write(json.dumps(entry)+"\n");self.log.flush()

    def step(self):
        self.contact.refresh()
        self.seam_contact.refresh()
        adhesive=self.velcro.refresh(time_s=self.step_count*self.dt,dt=self.dt)
        if adhesive["active_bonds"]!=self.last_bond_count:
            print(f"Velcro: {adhesive['active_bonds']} active / {adhesive['total_activated']} formed, t={self.step_count*self.dt:.2f}s",flush=True)
            self.last_bond_count=adhesive["active_bonds"]
            if adhesive["active_bonds"]:
                self.task.setdefault("adhesion_first_activated_phase",self.phase)
        self.scene.step();self.step_count+=1
        if self.release_min_bonds is not None:
            self.release_min_bonds=min(self.release_min_bonds,self.velcro.metrics()["active_bonds"])
        bp=array(self.bear.get_particles_pos());cp=array(self.cloth.get_particles_pos())
        if not np.isfinite(bp).all() or not np.isfinite(cp).all():raise RuntimeError("Nonfinite deformable state")
        drift=float(np.linalg.norm(bp[self.fixed]-self.initial["bear"][self.fixed],axis=1).max())
        self.max_base_drift=max(self.max_base_drift,drift)
        if drift>1e-6:raise RuntimeError(f"Bear fixed base moved {drift} m")
        contacts=self.robots["left"].get_contacts(with_entity=self.robots["right"])
        count=len(contacts["geom_a"]);self.max_arm_contacts=max(self.max_arm_contacts,count)
        if count:
            self.task["arm_collision"]={"step":self.step_count,"phase":self.phase,"positions":array(contacts["position"]).tolist()}
            raise RuntimeError("Inter-arm collision")
        for s,r in self.robots.items():
            self_contact=r.get_contacts(with_entity=r)
            self_count=len(self_contact["geom_a"])
            self.max_self_contacts=max(self.max_self_contacts,self_count)
            if self_count:
                self.task["self_collision"]={"side":s,"step":self.step_count,"phase":self.phase,
                    "link_a":[self.scene.rigid_solver.links[int(i)].name for i in array(self_contact["link_a"])],
                    "link_b":[self.scene.rigid_solver.links[int(i)].name for i in array(self_contact["link_b"])],
                    "position":array(self_contact["position"]).tolist(),"force_a":array(self_contact["force_a"]).tolist()}
                raise RuntimeError(f"{s} robot self-collision")
            table_count=len(r.get_contacts(with_entity=self.args.table_entity)["geom_a"])
            self.max_table_contacts=max(self.max_table_contacts,table_count)
            if table_count:raise RuntimeError(f"{s} robot contacted tabletop")
        stretch=np.linalg.norm(cp[self.cloth_edges[:,0]]-cp[self.cloth_edges[:,1]],axis=1)/self.rest_lengths
        self.max_cloth_stretch=max(self.max_cloth_stretch,float(stretch.max()))
        if stretch.max()>1.8:raise RuntimeError(f"Excessive cloth stretch {stretch.max():.3f}")
        if cp[:,2].min()<self.table-.005:raise RuntimeError("Cloth penetrated tabletop")
        for s,binding in self.bindings.items():
            h=self.hands[s]
            predicted=np.asarray(binding["local_positions"])@matrix(array(h.get_quat())).T+array(h.get_pos())
            error=float(np.linalg.norm(cp[binding["indices"]]-predicted,axis=1).max())
            self.max_grasp_error=max(self.max_grasp_error,error)
            if error>.025:raise RuntimeError(f"{s} grasp constraint drift {error} m")
        if self.step_count%20==0:self.record(bp,cp)
        if self.video and self.step_count*self.dt+1e-9>=self.next_frame:
            self.frame();self.next_frame+=1/self.args.record_fps
        if self.step_count%100==0:print(f"Step {self.step_count}, {self.phase}, max cloth stretch {stretch.max():.3f}",flush=True)

    def wait(self,seconds):
        for _ in range(round(seconds/self.dt)):self.step()

    def move(self,targets,seconds,quats=None,joint=False):
        start={s:self.tcp(s) for s in targets}
        qs={s:array(self.hands[s].get_quat()) for s in targets}
        qe={s:np.asarray(quats[s] if quats else self.quats[s]) for s in targets}
        interpolators={s:Slerp([0,1],Rotation.from_quat(np.stack([qs[s],qe[s]])[:,[1,2,3,0]])) for s in targets}
        n=round(seconds/self.dt)
        if joint:
            qstart={s:array(self.robots[s].get_qpos()) for s in targets}
            qend={s:self.solve(s,targets[s],qe[s]) for s in targets}
        for i in range(1,n+1):
            u=i/n;v=u*u*u*(10-15*u+6*u*u)
            if i==1 or i%2==0 or i==n:
                for s,target in targets.items():
                    if joint:q=qstart[s]+v*(qend[s]-qstart[s])
                    else:
                        qrot=interpolators[s]([v]).as_quat()[0][[3,0,1,2]]
                        q=self.solve(s,start[s]+v*(np.asarray(target)-start[s]),qrot)
                    q[7:]=self.grips[s]
                    self.robots[s].control_dofs_position(q)
            self.step()
        self.quats.update(qe)
        self.wait(.12)
        errors={s:float(np.linalg.norm(self.tcp(s)-targets[s])) for s in targets}
        if max(errors.values())>.015:raise RuntimeError(f"Tracking failed: {errors}")

    def pads(self,side):
        cp=array(self.cloth.get_particles_pos());groups=[];details=[]
        for finger in self.fingers[side]:
            local=(cp-array(finger.get_pos()))@matrix(array(finger.get_quat()))
            d=np.abs(local-[0,.0055,.0445])-[.0085,.004,.0085]
            dist=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)
            nearby=self.eligible[side][dist[self.eligible[side]]<=.004]
            nearby=nearby[np.argsort(dist[nearby])[:4]]
            groups.append(nearby)
            details.append({"finger":finger.name,"indices":nearby.tolist(),"distances_m":dist[nearby].tolist(),
                "closest_corner_patch_distance_m":float(dist[self.eligible[side]].min())})
        return groups,details

    def move_planned(self,name,seconds):
        """Track the collision-audited IK branch with motor targets, never qpos setters."""
        samples=[p for p in self.plan["samples"]+self.velcro_plan["samples"] if p["stage"]==name]
        if not samples:raise RuntimeError(f"Missing planned trajectory: {name}")
        coordinates=np.asarray([0.]+[p["u"] for p in samples])
        joints={s:np.vstack([array(r.get_qpos())[:7]]+[p["joints"][s] for p in samples]) for s,r in self.robots.items()}
        n=round(seconds/self.dt)
        for i in range(1,n+1):
            u=i/n;v=u*u*u*(10-15*u+6*u*u)
            for s,r in self.robots.items():
                q=np.r_[[np.interp(v,coordinates,joints[s][:,j]) for j in range(7)],self.grips[s]]
                r.control_dofs_position(q)
            self.step()
            if self.phase=="press" and self.seam_ready():
                for r in self.robots.values():r.control_dofs_position(array(r.get_qpos()))
                self.quats={s:array(h.get_quat()) for s,h in self.hands.items()}
                self.wait(.4)
                if self.seam_ready():
                    self.seam_complete=True
                    self.task["press_contact_stop"]={"planned_stage":name,"fraction":v,
                        "time_s":self.step_count*self.dt,"actual_tcp_m":{s:self.tcp(s).tolist() for s in self.robots},
                        "reason":"Stop motor trajectory once actual contact establishes an ideal local adhesive attachment that persists during a stationary dwell."}
                    return
        self.quats={s:np.asarray(q) for s,q in samples[-1]["quat_wxyz"].items()}
        self.wait(.18)
        errors={s:float(np.linalg.norm(self.tcp(s)-samples[-1]["tcp"][s])) for s in self.robots}
        if max(errors.values())>.015:raise RuntimeError(f"Planned tracking failed at {name}: {errors}")

    def grasp(self):
        self.start("pregrasp")
        cp=array(self.cloth.get_particles_pos())
        targets={s:cp[i]+[0,0,.198] for s,i in self.corner.items()}
        down={s:np.array([0,1,0,0]) for s in self.robots}
        self.move_planned("pregrasp",1.6)
        self.start("grasp")
        targets={s:array(self.cloth.get_particles_pos())[i]+self.grasp_inset[s]+[0,0,.024] for s,i in self.corner.items()}
        self.move(targets,1.4,down)
        for s,r in self.robots.items():
            self.grips[s]=np.array([.001,.001]);q=array(r.get_qpos());q[7:]=self.grips[s];r.control_dofs_position(q)
        self.wait(.3)
        for iteration in range(140):
            for s,r in self.robots.items():
                if s in self.bindings:continue
                cp=array(self.cloth.get_particles_pos())
                target=cp[self.corner[s]].copy()+self.grasp_inset[s];target[2]=max(self.table+.011,target[2]+.009)
                current=self.tcp(s);goal=current+np.clip(target-current,-.001,.001)
                r.control_dofs_position(self.solve(s,goal,down[s]))
            self.step()
            for s in self.robots:
                if s in self.bindings:continue
                groups,details=self.pads(s)
                if all(len(g) for g in groups) and array(self.robots[s].get_qpos())[7:].max()<.004:
                    indices=np.unique(np.concatenate(groups))
                    if self.corner[s] not in indices:
                        continue
                    rest_patch=self.data["cotton"]["points"][indices,:2]
                    if self.config.get("grasp",{}).get("require_surface_patch",False) and np.linalg.matrix_rank(rest_patch-rest_patch.mean(0),tol=1e-6)<2:
                        continue
                    cp=array(self.cloth.get_particles_pos());h=self.hands[s]
                    local=(cp[indices]-array(h.get_pos()))@matrix(array(h.get_quat()))
                    self.cloth.fix_particles_to_link(h.idx,particles_idx_local=indices)
                    self.bindings[s]={"indices":indices.tolist(),"local_positions":local.tolist()}
                    self.task["grasps"][s]={"time_s":self.step_count*self.dt,"pad_contacts":details,
                        "attached_indices":indices.tolist(),"original_corner_included":True}
                    self.robots[s].control_dofs_position(array(self.robots[s].get_qpos()))
                    print(f"{s} grasp gated by both pads: {indices.tolist()}",flush=True)
            if len(self.bindings)==2:break
        if len(self.bindings)!=2:
            self.task["failed_pad_contacts"]={s:self.pads(s)[1] for s in self.robots if s not in self.bindings}
            raise RuntimeError("Failed contact gate at both front cloth corners")
        self.wait(.3);self.save("grasp")

    def save(self,name):
        previous=self.run/f'state-{name}.npz'
        if previous.exists():
            with np.load(previous,allow_pickle=False) as old:
                previous_step=json.loads(str(old['metadata']))['step']
            history=self.run/'checkpoint-history'/f'{name}-{previous_step:06d}'
            history.mkdir(parents=True,exist_ok=True)
            import shutil
            shutil.copy2(previous,history/previous.name)
            for old_image in self.run.glob(name+'-*.png'):shutil.copy2(old_image,history/old_image.name)
        state={"bear_positions":array(self.bear.get_particles_pos()),"bear_velocities":array(self.bear.get_particles_vel()),
            "cloth_positions":array(self.cloth.get_particles_pos()),"cloth_velocities":array(self.cloth.get_particles_vel()),
            **{s+"_qpos":array(r.get_qpos()) for s,r in self.robots.items()}}
        span=slice(self.cloth.particle_start,self.cloth.particle_end)
        state["cloth_free_flags"]=self.scene.sim.pbd_solver.particles.free.to_numpy()[span]
        state["cloth_attached_link_indices"]=self.scene.sim.coupler.particle_attach_info.link_idx.to_numpy()[span]
        if name in ("press","release","hold","failure"):
            for s in self.robots:
                vertices=[];faces=[];offset=0
                for link in [self.hands[s],*self.fingers[s]]:
                    for geom in link.geoms:
                        verts=array(geom.get_verts())
                        vertices.append(verts);faces.append(np.asarray(geom.get_trimesh().faces)+offset)
                        offset+=len(verts)
                state[s+"_hand_collision_vertices"]=np.vstack(vertices)
                state[s+"_hand_collision_faces"]=np.vstack(faces)
        meta={"phase":name,"bindings":self.bindings,"grips":{s:v.tolist() for s,v in self.grips.items()},
            "quats":{s:v.tolist() for s,v in self.quats.items()},"step":self.step_count,
            "config":self.config,"asset_sha256":self.report["source_sha256"],"grasps":self.task["grasps"]}
        meta["velcro"]=self.velcro.dump()
        meta["seam_surface_contact"]=self.seam_contact.metrics(positions=state["cloth_positions"])
        if self.seam_contact.cache is not None:
            meta["seam_contact_cache"]={key:self.seam_contact.cache[key].tolist()
                for key in ("selected","target_face_ids","sides")}
        meta["release"]=self.task.get("release")
        meta["release_min_bonds"]=self.release_min_bonds
        np.savez_compressed(self.run/f"state-{name}.npz",**state,metadata=json.dumps(meta))
        for camera_name in (() if self.args.no_render else ("overview","front","overhead","back","table_left","bear_rear")):
            rgb,*_=self.cameras[camera_name].render()
            Image.fromarray(np.asarray(rgb)).save(self.run/f"{name}-{camera_name}.png")
        self.persist()

    def restore(self,path):
        with np.load(path,allow_pickle=False) as a:
            meta=json.loads(str(a["metadata"]))
            if meta["config"]!=self.config or meta["asset_sha256"]!=self.report["source_sha256"]:
                raise RuntimeError("Resume requires identical config and assets")
            for s,r in self.robots.items():
                r.set_qpos(a[s+"_qpos"]);r.control_dofs_position(a[s+"_qpos"])
            for n,e in (("bear",self.bear),("cloth",self.cloth)):
                e.set_particles_pos(a[n+"_positions"]);e.set_particles_vel(a[n+"_velocities"])
        self.bindings=meta["bindings"]
        self.grips={s:np.array(q) for s,q in meta["grips"].items()}
        self.quats={s:np.array(q) for s,q in meta["quats"].items()}
        for s,b in self.bindings.items():
            self.cloth.fix_particles_to_link(self.hands[s].idx,particles_idx_local=b["indices"])
            h=self.hands[s]
            cp=array(self.cloth.get_particles_pos())
            b["local_positions"]=((cp[b["indices"]]-array(h.get_pos()))@matrix(array(h.get_quat()))).tolist()
        self.task["grasps"]=meta["grasps"]
        if "velcro" in meta:self.velcro.restore(meta["velcro"])
        self.seam_contact.reset_cache()
        if meta.get("release") is not None:self.task["release"]=meta["release"]
        self.release_min_bonds=meta.get("release_min_bonds")
        self.step_count=int(meta["step"])
        self.next_frame=self.step_count*self.dt
        self.task["initialization"]={"snapshot":str(path),"initialized_from_stage":meta["phase"],
            "state_setters_only_at_initialization":True}
        return meta["phase"]

    def execute(self):
        stage=self.restore(self.args.resume) if self.args.resume else None
        if stage is None:
            self.start("settle");self.wait(min(max(self.args.steps*self.dt,.3),3.))
            self.save("initial")
            self.grasp();stage="grasp"
        if self.args.stop_after==stage:return self.evaluate(partial=True)
        if stage=="grasp":
            self.start("lift")
            targets={s:self.tcp(s) for s in self.robots}
            for v in targets.values():v[2]=1.05
            self.move_planned("lift",2.0)
            self.wait(.4);self.save("lift");stage="lift"
        if self.args.stop_after==stage:return self.evaluate(partial=True)
        if stage=="lift":
            self.start("approach_body")
            wp=next(p for p in self.plan["waypoints"] if p["stage"]=="taut_front")
            self.move_planned("taut_front",2.8);self.wait(.3);self.save("approach_body");stage="approach_body"
        if self.args.stop_after==stage:return self.evaluate(partial=True)
        wrap_names=[p["stage"] for p in self.plan["waypoints"] if p["stage"].startswith(("taut_","cross_")) and p["stage"]!="taut_front"]
        if stage=="approach_body" or stage in wrap_names:
            self.start("wrap")
            waypoints=[p for p in self.plan["waypoints"] if p["stage"].startswith(("taut_","cross_")) and p["stage"]!="taut_front"]
            if stage in wrap_names:waypoints=waypoints[wrap_names.index(stage)+1:]
            for j,wp in enumerate(waypoints):
                self.move_planned(wp["stage"],wp.get("seconds",1.6 if wp["stage"]=="cross_90" else 1.0))
                self.save(wp["stage"])
            self.save("wrap");stage="wrap"
        if self.args.stop_after==stage:return self.evaluate(partial=True)
        press_names=[p["stage"] for p in self.velcro_plan["waypoints"] if p["stage"].startswith("press")]
        if stage=="wrap" or stage in press_names:
            self.start("press")
            if stage=="wrap":
                self.refine_seam_current_pose()
                if self.seam_ready():self.seam_complete=True
            remaining=[] if getattr(self,"seam_complete",False) else [p for p in self.velcro_plan["waypoints"] if p["stage"] in press_names]
            if stage in press_names:remaining=remaining[press_names.index(stage)+1:]
            for wp in remaining:
                self.move_planned(wp["stage"],wp.get("seconds",1.5))
                if wp["stage"]=="press_separate" and not getattr(self,"seam_complete",False):
                    self.refine_seam_current_pose()
                    if self.seam_ready():self.seam_complete=True
                    elif self.velcro.metrics()['active_bonds']:
                        # Once a real bond exists, moving along the old open-loop
                        # alignment path would tear the established seam.
                        self.refine_seam_current_pose()
                        if self.seam_ready():self.seam_complete=True
                        else:raise RuntimeError('Existing partial seam requires further contact alignment; open-loop press path stopped')
                if getattr(self,"seam_complete",False):
                    self.save("press_contact_stop");break
                self.save(wp["stage"])
            self.wait(.6);self.tighten_seam();self.save("press");stage="press"
            if not self.seam_ready():
                raise RuntimeError("The strips did not establish a contact-gated adhesive attachment")
        if self.args.stop_after==stage:return self.evaluate(partial=True)
        if stage=="press":
            if not self.seam_ready():
                raise RuntimeError("Cannot release grippers without an established adhesive seam")
            self.start("release")
            self.velcro.enabled=False
            self.release_min_bonds=self.velcro.metrics()["active_bonds"]
            self.task["release"]={"time_s":self.step_count*self.dt,"active_bonds_before_release":self.velcro.metrics()["active_bonds"],
                "released_particles":{},"release_times_s":{},
                "surviving_bond_ids_before_release":[b["id"] for b in self.velcro.metrics()["activation_events"] if b["active"]],
                "new_adhesive_activation_disabled":True,"tcp_before_retreat":{s:self.tcp(s).tolist() for s in self.robots}}
            for wp in self.velcro_plan["waypoints"]:
                if wp["stage"].startswith("retreat"):
                    # A partially opened gripper can first clear the bear,
                    # then open fully at the independently checked next pose.
                    for s, opening in wp.get("gripper_openings_before",{}).items():
                        if s in self.bindings or not .001<=opening<=.04:
                            raise RuntimeError("Invalid staged gripper opening")
                        self.grips[s]=np.array([opening,opening])
                        q=array(self.robots[s].get_qpos());q[7:]=self.grips[s]
                        self.robots[s].control_dofs_position(q)
                        self.wait(.25)
                    for s in wp.get("release_sides_before",[]):
                        if not self.seam_ready():
                            raise RuntimeError("Adhesion failed during sequential gripper release")
                        b=self.bindings.pop(s)
                        self.cloth.release_particle(particles_idx_local=b["indices"])
                        self.task["release"]["released_particles"][s]=b["indices"]
                        self.task["release"]["release_times_s"][s]=self.step_count*self.dt
                        opening=wp.get("release_finger_opening_m",{}).get(s,.04)
                        if not .001<=opening<=.04:
                            raise RuntimeError("Invalid release finger opening")
                        self.task["release"].setdefault("finger_openings_at_release_m",{})[s]=opening
                        self.grips[s]=np.array([opening,opening])
                        q=array(self.robots[s].get_qpos());q[7:]=self.grips[s]
                        self.robots[s].control_dofs_position(q)
                        self.wait(.25)
                    self.move_planned(wp["stage"],wp.get("seconds",1.5))
            if self.bindings:raise RuntimeError("Retreat plan did not release both grippers")
            self.save("release");stage="release"
        if self.args.stop_after==stage:return self.evaluate(partial=True)
        self.start("hold");self.wait(2.0);self.save("hold")
        return self.evaluate(partial=False)

    def evaluate(self,partial=False):
        bp=array(self.bear.get_particles_pos());cp=array(self.cloth.get_particles_pos())
        pen=self.contact.validate(bp,cp,full=True)
        center=np.asarray(self.config["bear"]["position_xy"])
        path=cp[self.front_edge,:2]-center
        angles=np.unwrap(np.arctan2(path[:,1],path[:,0]))
        arc=float(abs(angles[-1]-angles[0]))
        closed=np.vstack([path,path[0]])
        cross=closed[:-1,0]*closed[1:,1]-closed[:-1,1]*closed[1:,0]
        winding=float(np.arctan2(cross,(closed[:-1]*closed[1:]).sum(1)).sum()/(2*np.pi))
        endpoint_gap=float(np.linalg.norm(cp[self.front_edge[0]]-cp[self.front_edge[-1]]))
        stretch=np.linalg.norm(cp[self.cloth_edges[:,0]]-cp[self.cloth_edges[:,1]],axis=1)/self.rest_lengths
        triangles=cp[self.data["cotton"]["faces"]]
        centroids=triangles.mean(1)
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        _,distance,_=trimesh.proximity.closest_point(self.contact.geometry.mesh(bp),centroids)
        torso=(centroids[:,2]>self.table+.10)&(centroids[:,2]<self.table+.30)
        close=torso&(distance<.015)
        angle=np.degrees(np.arctan2(centroids[:,0]-center[0],-(centroids[:,1]-center[1])))
        sectors=np.floor(((angle+45)%360)/90).astype(int)
        contact_area={name:float(areas[close&(sectors==i)].sum()) for i,name in enumerate(("front","right","back","left"))}
        near_area=float(areas[close].sum())
        self.task["metrics"]={"front_edge_angular_span_deg":float(np.degrees(arc)),
            "front_edge_closed_winding_number":winding,"held_corner_gap_m":endpoint_gap,
            "cloth_min_height_above_table_m":float(cp[:,2].min()-self.table),
            "cloth_max_stretch_ratio_final":float(stretch.max()),"cloth_max_stretch_ratio_over_run":self.max_cloth_stretch,
            "bear_base_max_displacement_m":self.max_base_drift,"inter_arm_contacts":self.max_arm_contacts,
            "robot_self_contacts":self.max_self_contacts,
            "robot_table_contacts":self.max_table_contacts,
            "max_grasp_constraint_error_m":self.max_grasp_error,"penetration":pen}
        self.task["metrics"]["torso_contact"]={"band_height_above_table_m":[.10,.30],"near_distance_m":.015,
            "near_surface_cloth_area_m2":near_area,"sector_area_m2":contact_area,
            "cloth_area_fraction_near_torso":float(near_area/areas.sum())}
        checks={"both_front_corners_contact_grasped":len(self.task["grasps"])==2,
            "bear_base_fixed":self.max_base_drift<=1e-6,"no_inter_arm_collision":self.max_arm_contacts==0,
            "no_robot_self_collision":self.max_self_contacts==0,
            "no_robot_table_collision":self.max_table_contacts==0,
            "cloth_not_overstretched":stretch.max()<1.3,"finite_state":bool(np.isfinite(cp).all() and np.isfinite(bp).all()),
            "no_deep_cloth_body_penetration":pen["max_penetration_m"]<.003,
            "bear_stays_upright":.85<float(np.ptp(bp[:,2])/np.ptp(self.initial["bear"][:,2]))<1.15}
        if not partial or self.phase in ("press","thread_arms"):
            checks.update(angular_wrap_over_300_deg=arc>=np.deg2rad(300),
                encircles_body=abs(winding)>.9,corners_near_behind_body=endpoint_gap<.14 and min(cp[self.front_edge[[0,-1]],1])>center[1],
                cloth_lifted_to_belly=cp[:,2].min()>self.table+.065,
                cloth_contacts_front_and_sides=near_area>.025 and all(contact_area[s]>.0015 for s in ("front","left","right")))
            adhesive=self.velcro.metrics()
            checks.update(velcro_joined_after_contact=adhesive["active_bonds"]>=1,
                adhesive_contact_patch_present=self.seam_ready(),
                adhesive_error_below_detachment_threshold=adhesive["max_bond_error_m"]<self.config["velcro"]["break_error_m"],
                activation_requires_actual_contact=bool(adhesive["activation_events"]) and all(b["activation_distance_m"]<=self.config["velcro"]["activation_distance_m"]+1e-7 and b["dwell_s"]>=self.config["velcro"]["dwell_time_s"]-1e-7
                    and b["activation_normal_dot"]<=-np.cos(np.deg2rad(self.config["velcro"]["normal_angle_tolerance_deg"]))+1e-7
                    and min(b["source_facing_gap_m"],b["target_facing_gap_m"])>=-self.config["velcro"]["facing_tolerance_m"]-1e-7 for b in adhesive["activation_events"]))
        if not partial:
            checks.update(velcro_joined_after_contact=self.velcro.metrics()["active_bonds"]>=1,
                both_grippers_released=len(self.bindings)==0 and "release" in self.task,
                adhesion_survives_without_gripper_attachments=self.velcro.metrics()["active_bonds"]>=1 and len(self.bindings)==0)
            adhesive=self.velcro.metrics()
            cloth_links=self.scene.sim.coupler.particle_attach_info.link_idx.to_numpy()[self.cloth.particle_start:self.cloth.particle_end]
            free=self.scene.sim.pbd_solver.particles.free.to_numpy()[self.cloth.particle_start:self.cloth.particle_end]
            self.task["release_verification"]={"cloth_nodes_attached_to_robot":int(np.sum(cloth_links>=0)),
                "cloth_nodes_fixed_in_world":int(np.sum(~free.astype(bool))),"held_without_grippers_s":2.0,
                "minimum_active_bonds_since_release":self.release_min_bonds}
            clearance={}
            samples=self.contact.geometry.samples(cp)
            for s in self.robots:
                distances=[]
                for link in [self.hands[s],*self.fingers[s]]:
                    for geom in link.geoms:
                        mesh=trimesh.Trimesh(array(geom.get_verts()),geom.get_trimesh().faces,process=False)
                        _,d,_=trimesh.proximity.closest_point(mesh,samples)
                        distances.append(float(d.min()))
                clearance[s]=min(distances)
            self.task["release_verification"]["hand_to_cloth_sample_clearance_m"]=clearance
            checks.update(no_cloth_world_or_robot_pins=bool(np.all(cloth_links<0) and np.all(free)),
                adhesive_contact_patch_present=self.seam_ready(),
                original_bonds_survive_release=self.release_min_bonds is not None and self.release_min_bonds>=6 and not self.velcro.enabled,
                grippers_clearly_away_from_cloth=min(clearance.values())>.03,
                adhesive_error_below_detachment_threshold=adhesive["max_bond_error_m"]<self.config["velcro"]["break_error_m"],
                activation_requires_actual_contact=all(b["activation_distance_m"]<=self.config["velcro"]["activation_distance_m"]+1e-7 and b["dwell_s"]>=self.config["velcro"]["dwell_time_s"]-1e-7
                    and b["activation_normal_dot"]<=-np.cos(np.deg2rad(self.config["velcro"]["normal_angle_tolerance_deg"]))+1e-7
                    and min(b["source_facing_gap_m"],b["target_facing_gap_m"])>=-self.config["velcro"]["facing_tolerance_m"]-1e-7 for b in adhesive["activation_events"]))
        checks={key:bool(value) for key,value in checks.items()}
        self.task["checks"]=checks
        self.report["status"]=("press_stage_passed" if self.phase=="press" else "partial_stage_passed") if partial and all(checks.values()) else "passed" if all(checks.values()) else "failed"
        self.persist()
        return all(checks.values())

    def close_videos(self):
        if self.replay_frames:
            np.savez_compressed(self.run/'replay-frames.npz',**{k:np.asarray([f[k] for f in self.replay_frames]) for k in self.replay_frames[0]})
            self.replay_frames=[]
        for name in ("video","left_video","rear_video"):
            encoder=getattr(self,name)
            if encoder is not None:
                setattr(self,name,None)
                encoder.close()

    def close(self):
        self.close_videos()
        self.log.close();self.velcro.close();self.seam_contact.close();self.contact.close()

def run_wrap(args):
    import genesis as gs
    from scene import read_assets,build_scene
    config=tomllib.loads(args.config.read_text())
    config["cameras"]["overhead"]={"position":[0.,-.04,2.25],"target":[0.,.02,.88],"fov":47.}
    config["cameras"]["back"]={"position":[.25,.95,1.25],"target":[0.,.12,1.00],"fov":42.}
    config["cameras"].setdefault("table_left",{"position":[-2.05,-.50,1.60],"target":[0.,-.02,1.00],"fov":43.})
    run=args.output or HERE/"runs"/(datetime.now().strftime("%Y%m%d-%H%M%S")+f"-wrap-{os.getpid()}")
    run.mkdir(parents=True,exist_ok=False)
    report={"status":"starting","config":config,"geometry":json.loads((HERE/"assets/metadata.json").read_text()),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/"assets").iterdir() if p.is_file()},
        "implementation_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.suffix in (".py",".toml",".json",".sh")},
        "scope":("Independent hook-and-loop variant: same environment and cloth, grasp through contact-triggered seam; both grippers remain holding at press endpoint, release not verified."
            if args.stop_after=="press" else "Independent hook-and-loop variant: same environment and cloth, contact-triggered seam, then release and verify.")}
    experiment=None;started=time.monotonic()
    try:
        data,mesh_dir=read_assets(config,run)
        scene,robots,bear,cloth,cameras,fixed,initial=build_scene(config,data,mesh_dir,args,report)
        from arms_task import ArmsExperiment
        experiment=ArmsExperiment(scene,robots,bear,cloth,cameras,fixed,initial,data,config,report,run,args)
        okay=experiment.execute()
        report["wall_seconds"]=time.monotonic()-started;experiment.persist()
        if not okay:raise RuntimeError(f"Wrapping checks not satisfied: {report['wrapping']['checks']}")
        print(f"PASS: {report['status']} {run}",flush=True)
        experiment.close_videos()
        if not args.headless:
            while scene.viewer.is_alive():scene.viewer.update();time.sleep(1/30)
    except Exception as exc:
        report.update(status="failed",error=str(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started)
        if experiment:
            try:experiment.save("failure")
            except Exception:pass
        (run/"report.json").write_text(json.dumps(report,indent=2))
        raise
    finally:
        if experiment:experiment.close()
        if gs._initialized:gs.destroy()
