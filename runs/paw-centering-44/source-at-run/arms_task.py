"""Two physical cloth apertures and contact-gated extraction of soft bear arms."""
import json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from wrap_task import WrapExperiment,array,matrix,PHASES
from arm_geometry import ArmMaterialCurve,find_hole_boundary,aperture_measurement,point_in_polygon

PHASES.update(regrasp_left="左机械臂换抓左侧熊手臂",pull_left="从左侧布孔拉出熊手臂",
              regrasp_right="右机械臂换抓右侧熊手臂",pull_right="从右侧布孔拉出熊手臂",
              thread_arms="双侧手臂已穿出布孔")

class ArmsExperiment(WrapExperiment):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.arm_bindings={}
        self.normal_control_sign={s:1. for s in self.robots}
        from robot_geometry import Planner
        self.clearance_planner=Planner()
        self.arm_curves={s:ArmMaterialCurve(self.data['bear']['points'],self.data['bear']['tetra'],s,
            [*self.config['bear']['position_xy'],self.table+self.config['bear']['base_clearance']],
            self.report['geometry']['bear'].get('arm_centerline_centers_local_m')) for s in self.robots}
        material_points=self.data['cotton']['points'].copy()
        material_points[:,:2]-=self.config['cloth']['position_xy']
        self.hole_boundaries={s:find_hole_boundary(material_points,self.data['cotton']['faces'],
            self.report['geometry']['cotton']['holes'][s]['center_material_xy_m']) for s in self.robots}
        self.task['completion_scope']='Wrap, join complementary Velcro strips, then sequentially grasp and draw both soft bear arms through real cloth openings.'
        self.report['scope']=self.task['completion_scope']
        self.task['arm_extraction']={'grasps':{},'measurements':[],
            'minimum_joined_patch':{'bonds':1,
                'scope':'A contact-gated ideal point attachment represents initial local hook engagement. Additional bonds require independent real contact. No full-width adhesive coverage, torsional stiffness, or calibrated load/peel rating is assumed.'},
            'arm_topology':'single connected soft tetrahedral bear',
            'grasp_model':'contact-gated local no-slip link constraints; no force closure calibration',
            'cloth_holes':'missing mesh faces, not a transparent texture'}

    def hole(self,side):
        cp=array(self.cloth.get_particles_pos())
        ids=self.hole_boundaries[side]
        points=cp[ids];center=points.mean(0)
        _,_,vh=np.linalg.svd(points-center)
        n=vh[-1];sign=-1 if side=='left' else 1
        if n[0]*sign<0:n=-n
        return center,n,points

    def paw(self,side):
        return array(self.bear.get_particles_pos())[self.data['bear'][side+'_paw_indices']].mean(0)

    def record_arm_geometry(self,label):
        row={'label':label,'time_s':self.step_count*self.dt,'sides':{}}
        for s in self.robots:
            c,n,p=self.hole(s);paw=self.paw(s)
            cp=array(self.cloth.get_particles_pos());bp=array(self.bear.get_particles_pos())
            row['sides'][s]={'hole_center_m':c.tolist(),'hole_normal':n.tolist(),'paw_center_m':paw.tolist(),
                'paw_signed_distance_to_hole_m':float(np.dot(paw-c,n)),
                'paw_min_signed_distance_to_hole_m':float(((bp[self.data['bear'][s+'_paw_indices']]-c)@n).min()),
                'hole_min_boundary_radius_m':float(np.linalg.norm(p-c,axis=1).min()),
                'material_curve':aperture_measurement(self.arm_curves[s].positions(bp),cp[self.hole_boundaries[s]],s)}
        self.task['arm_extraction']['measurements'].append(row)
        return row

    def record(self,bp,cp):
        super().record(bp,cp)
        if self.phase.startswith(('regrasp','pull')) or self.phase=='thread_arms':
            row=self.record_arm_geometry('trajectory')
            self.log.write(json.dumps({'arm_geometry':row})+'\n');self.log.flush()

    def step(self):
        super().step()
        if self.phase.startswith(('regrasp','pull')) or self.phase=='thread_arms':
            active=self.velcro.metrics()['active_bonds']
            self.task['arm_extraction']['minimum_active_bonds']=min(active,self.task['arm_extraction'].get('minimum_active_bonds',active))
            if not active:raise RuntimeError('Velcro seam detached during arm extraction')
        for side,binding in self.arm_bindings.items():
            h=self.hands[side]
            target=np.asarray(binding['local_positions'])@matrix(array(h.get_quat())).T+array(h.get_pos())
            bp=array(self.bear.get_particles_pos())
            error=float(np.linalg.norm(bp[binding['indices']]-target,axis=1).max())
            self.task['arm_extraction']['max_grasp_error_m']=max(error,self.task['arm_extraction'].get('max_grasp_error_m',0.))
            if error>.015:raise RuntimeError('Soft arm grasp constraint drifted beyond 15 mm')

    def arm_pad_contacts(self,side):
        bp=array(self.bear.get_particles_pos())
        eligible=self.data['bear'][side+'_paw_indices']
        groups=[];details=[]
        for finger in self.fingers[side]:
            local=(bp-array(finger.get_pos()))@matrix(array(finger.get_quat()))
            delta=np.abs(local-[0,.0055,.0445])-[.0085,.004,.0085]
            distance=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)
            near=eligible[distance[eligible]<=.005]
            near=near[np.argsort(distance[near])[:4]]
            groups.append(near)
            details.append({'finger':finger.name,'indices':near.tolist(),'distance_m':distance[near].tolist(),
                            'closest_distance_m':float(distance[eligible].min())})
        return groups,details

    def tighten_seam(self):
        self.task.setdefault('seam_alignment',[])
        if self.seam_ready():return
        self.refine_seam_current_pose()
        if self.seam_ready():return
        reference=Rotation.from_quat(np.array(self.velcro_plan['waypoints'][-1]['quat_wxyz']['left'])[[1,2,3,0]])
        lifted=(Rotation.from_rotvec([np.deg2rad(80),0,0])*reference).as_quat()[[3,0,1,2]]
        if (Rotation.from_quat(lifted[[1,2,3,0]])*Rotation.from_quat(array(self.hands['left'].get_quat())[[1,2,3,0]]).inv()).magnitude()>.03:
            self.move({'left':self.tcp('left')},1.8,{'left':lifted})
            self.save('press_clear_wrist')
        for attempt,gap in enumerate((.008,.005)):
            if self.seam_ready():return
            target={s:self.tcp(s) for s in self.robots}
            target['left'][0]=-gap/2;target['right'][0]=gap/2
            self.move(target,1.2)
            self.wait(.8)
            self.task['seam_alignment'].append({'attempt':attempt,'time_s':self.step_count*self.dt,
                'metrics':self.velcro.metrics()})
            self.save('press_alignment_'+str(attempt))
        self.refine_seam_current_pose()

    def refine_seam_current_pose(self):
        self.task.setdefault('seam_alignment',[])
        initial_tips={s:self.tcp(s).copy() for s in self.robots}
        for attempt in range(32):
            if self.seam_ready():return
            cp=array(self.cloth.get_particles_pos());g=self.velcro.geometry.candidates(cp)
            bonded=[b['source_node'] for b in self.velcro.metrics()['activation_events'] if b['active']]
            valid=(g['normal_dot']<0)&(g['source_facing_gap_m']>=-.001)&(g['target_facing_gap_m']>=-.001)&~np.isin(g['source_nodes'],bonded)
            if not valid.any():break
            angle_error=np.maximum(0,np.arccos(np.clip(-g['normal_dot'],-1,1))-np.deg2rad(35))
            cost=g['distance_m']+.03*angle_error
            index=np.flatnonzero(valid)[np.argmin(cost[valid])]
            print(f"Seam refine {attempt}: gap={g['distance_m'][index]:.4f} m, normal dot={g['normal_dot'][index]:.3f}, bonds={len(bonded)}",flush=True)
            current=Rotation.from_quat(array(self.hands['left'].get_quat())[[1,2,3,0]])
            if g['normal_dot'][index]>-np.cos(np.deg2rad(35)):
                source=g['target_active_normals'][index];desired=-g['source_active_normals'][index]
                axis=np.cross(source,desired);axis/=max(np.linalg.norm(axis),1e-12)
                error_angle=float(np.arccos(np.clip(np.dot(source,desired),-1,1)))
                angle=min(.08,max(.02,error_angle-np.deg2rad(33)))
                choice=None
                for side,turn_sign in [('right',1),('left',-1)]:
                    present=Rotation.from_quat(array(self.hands[side].get_quat())[[1,2,3,0]])
                    for fraction in (1.,.5):
                        candidate=Rotation.from_rotvec(axis*angle*turn_sign*fraction*self.normal_control_sign[side])*present
                        if self.press_pose_clear(side,self.tcp(side),candidate):choice=(side,candidate);break
                    if choice is not None:break
                if choice is None:break
                side,rotation=choice;target_quat=rotation.as_quat()[[3,0,1,2]]
                before_dot=float(g['normal_dot'][index])
                source_face=int(g['source_face_ids'][index]);target_face=int(g['target_face_ids'][index])
                self.move({side:self.tcp(side)},.55,{side:target_quat});self.wait(.15)
                from velcro_contact import face_frames
                points=array(self.cloth.get_particles_pos());faces=self.data['cotton']['faces']
                normals=face_frames(points[faces[[source_face,target_face]]])[:,:,2]
                after_dot=float(np.dot(normals[0],normals[1])*self.config['velcro']['surface_signs']['left']*self.config['velcro']['surface_signs']['right'])
                if after_dot>before_dot+.002:self.normal_control_sign[side]*=-1
                self.task['seam_alignment'].append({'normal_alignment_attempt':attempt,'moving_side':side,
                    'source_face_id':source_face,'target_face_id':target_face,'normal_dot_before':before_dot,'normal_dot_after':after_dot,
                    'next_rotation_feedback_sign':self.normal_control_sign[side],'time_s':self.step_count*self.dt})
                self.save('press_normal_'+str(attempt))
                continue
            target=(cp[g['target_nodes'][index]]*g['barycentric'][index,:,None]).sum(0)
            delta=target-cp[g['source_nodes'][index]];distance=np.linalg.norm(delta)
            if distance<.0035:
                self.wait(.5)
                continue
            delta*=min(.004,max(0.,distance-.003))/distance
            side='left';destination=self.tcp(side)+delta
            if np.linalg.norm(destination-initial_tips[side])>.04 or not self.press_pose_clear(side,destination,current):
                side='right';destination=self.tcp(side)-delta
                current=Rotation.from_quat(array(self.hands[side].get_quat())[[1,2,3,0]])
                if np.linalg.norm(destination-initial_tips[side])>.04 or not self.press_pose_clear(side,destination,current):break
            lifted=current.as_quat()[[3,0,1,2]]
            self.move({side:destination},.55,{side:lifted});self.wait(.15)
            self.task['seam_alignment'].append({'surface_guided_attempt':attempt,'time_s':self.step_count*self.dt,
                'measured_gap_m':float(distance),'moving_side':side,'motor_tcp_shift_m':(delta if side=='left' else -delta).tolist(),'metrics':self.velcro.metrics()})
            self.save('press_surface_'+str(attempt))

    def press_pose_clear(self,side,position,rotation):
        planner=self.clearance_planner
        joints={s:array(r.get_qpos())[:7] for s,r in self.robots.items()}
        q,error,angle=planner.solve(side,position,rotation,joints[side])
        if error>.002 or angle>.02:return False
        joints[side]=q;clearance=planner.clearance(joints)
        return clearance['arms']['distance']>.005 and clearance[side+'_bear']['distance']>.008 and clearance[side+'_table']['distance']>.01

    def seam_ready(self):
        m=self.velcro.metrics()
        return m['active_bonds']>=1 and m['max_bond_error_m']<self.config['velcro']['break_error_m']

    def restore(self,path):
        stage=super().restore(path)
        with np.load(path,allow_pickle=False) as state:meta=json.loads(str(state['metadata']))
        self.arm_bindings=meta.get('arm_bindings',{})
        for side,binding in self.arm_bindings.items():
            self.bear.fix_particles_to_link(self.hands[side].idx,particles_idx_local=binding['indices'])
        planned={wp['stage'] for wp in self.velcro_plan['waypoints']}
        if stage.startswith('press_') and stage not in planned:stage='press'
        if stage=='press_separate':
            self.start('press');self.refine_seam_current_pose()
            if self.seam_ready():
                self.save('press');return 'press'
        if stage=='press':
            self.start('press');self.tighten_seam();self.save('press')
        return stage

    def move(self,targets,seconds,quats=None,joint=False):
        if self.phase!='press' and not self.phase.startswith(('regrasp','pull')):
            return super().move(targets,seconds,quats,joint)
        # Plan a continuous bounded joint branch before commanding motors.
        # Random IK restarts near joint limits can produce discontinuous elbows.
        planner=self.clearance_planner
        start={s:self.tcp(s) for s in targets}
        initial={s:array(r.get_qpos())[:7] for s,r in self.robots.items()}
        qe={s:np.asarray(quats[s] if quats else array(self.hands[s].get_quat())) for s in targets}
        rotations={s:Slerp([0,1],Rotation.from_quat(np.stack([array(self.hands[s].get_quat()),qe[s]])[:,[1,2,3,0]])) for s in targets}
        samples={s:[initial[s].copy()] for s in targets};current={s:q.copy() for s,q in initial.items()}
        joint_end={}
        if joint:
            from robot_geometry import HOME
            for s in targets:
                q,e,a=planner.solve(s,np.asarray(targets[s]),rotations[s]([1])[0],HOME)
                if e>.003 or a>.04:raise RuntimeError('Staging endpoint unreachable')
                joint_end[s]=q
        for u in np.linspace(0,1,61)[1:]:
            for s in targets:
                if joint:
                    q=initial[s]+u*(joint_end[s]-initial[s]);e=0.;a=0.
                else:q,e,a=planner.solve(s,start[s]+u*(np.asarray(targets[s])-start[s]),rotations[s]([u])[0],current[s])
                if e>.003 or a>.04:raise RuntimeError(f'Continuous arm IK failed: {s}, u={u:.3f}, position={e}, angle={a}')
                if np.max(np.abs(q-current[s]))>.25:raise RuntimeError('Discontinuous arm IK branch')
                current[s]=q;samples[s].append(q)
            clearance=planner.clearance(current)
            if clearance['arms']['distance']<.003 or any(clearance[s+'_table']['distance']<.005 or clearance[s+'_self']['distance']<.003 for s in targets):
                raise RuntimeError(f'Arm route clearance failed at {u}: {clearance}')
        self.task['arm_extraction'].setdefault('motor_paths',[]).append({'side':list(targets),'phase':self.phase,'start_s':self.step_count*self.dt,'targets':{s:np.asarray(t).tolist() for s,t in targets.items()},'continuous_ik_samples':61})
        n=round(seconds/self.dt);coordinates=np.linspace(0,1,61)
        for i in range(1,n+1):
            u=i/n;v=u*u*u*(10-15*u+6*u*u)
            for s in targets:
                values=np.asarray(samples[s]);q=np.r_[[np.interp(v,coordinates,values[:,j]) for j in range(7)],self.grips[s]]
                self.robots[s].control_dofs_position(q)
            self.step()
        self.quats.update(qe);self.wait(.2)
        errors={s:float(np.linalg.norm(self.tcp(s)-targets[s])) for s in targets}
        if max(errors.values())>.015:raise RuntimeError(f'Continuous motor tracking failed: {errors}')

    def release_cloth(self,side):
        binding=self.bindings.pop(side)
        self.cloth.release_particle(particles_idx_local=binding['indices'])
        indices=np.asarray(binding['indices'])+self.cloth.particle_start
        links=self.scene.sim.coupler.particle_attach_info.link_idx.to_numpy()[indices]
        if np.any(links>=0):raise RuntimeError('Released cloth retained link attachments')
        self.task['arm_extraction'].setdefault('cloth_release_events',[]).append(
            {'side':side,'time_s':self.step_count*self.dt,'indices':binding['indices'],
             'active_bonds':self.velcro.metrics()['active_bonds']})
        self.grips[side]=np.array([.022,.022])
        q=array(self.robots[side].get_qpos());q[7:]=self.grips[side]
        self.robots[side].control_dofs_position(q)
        self.wait(.3)

    def free_hand(self,side):
        sign=-1 if side=='left' else 1
        self.start('regrasp_'+side)
        self.release_cloth(side)
        self.move({side:self.tcp(side)+[sign*.115,0,.020]},2.0)
        quat=np.array([0.,1.,0.,0.])
        self.grips[side]=np.array([.035,.035])
        self.move({side:[sign*.34,.15,1.10]},3.0,{side:quat},joint=True)
        self.save('regrasp_'+side+'_outside')

    def extract(self,side,skip_to=None):
        sign=-1 if side=='left' else 1
        if skip_to=='pull':return self.pull_arm(side)
        self.start('regrasp_'+side)
        if side in self.bindings:self.free_hand(side)
        self.record_arm_geometry('before_grasp_'+side)
        center,normal,boundary=self.hole(side)
        paw=self.paw(side)
        target=paw.copy();target[2]=np.clip(center[2],paw[2]-.012,paw[2]+.012)
        # Enter along the aperture normal, not from its centroid toward an
        # elevated paw: the latter can put the palm inside the torso.
        direction=-normal
        jaw=np.array([0.,1.,0.]);jaw-=direction*np.dot(jaw,direction);jaw/=np.linalg.norm(jaw)
        base_rotation=Rotation.from_matrix(np.column_stack([np.cross(jaw,direction),jaw,direction]))
        entry=target-normal*np.dot(target-center,normal)
        approach=entry+normal*.085
        chosen=None
        _,_,plane=np.linalg.svd(boundary-center);polygon=(boundary-center)@plane[:2].T
        planner=self.clearance_planner;current={s:array(r.get_qpos())[:7] for s,r in self.robots.items()}
        zvalues=sorted(set([float(target[2]),float(paw[2]),float(paw[2]-.012),float(paw[2]+.012)]),key=lambda z:abs(z-center[2]))
        for z in zvalues:
            candidate_target=paw.copy();candidate_target[2]=z
            entry=candidate_target-normal*np.dot(candidate_target-center,normal)
            candidate_approach=entry+normal*.085
            for roll in (0.,np.pi,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4):
                orientation=base_rotation*Rotation.from_rotvec([0,0,roll])
                jaw_axis=orientation.as_matrix()[:,1]
                footprint=[entry+v*jaw_axis for v in (-.020,0.,.020)]
                if not all(point_in_polygon((v-center)@plane[:2].T,polygon)[0] for v in footprint):continue
                from robot_geometry import HOME
                qpre,e,angle=planner.solve(side,candidate_approach,orientation,HOME)
                if e>.003 or angle>.04:continue
                valid=True
                for u in np.linspace(0,1,61):
                    q=current[side]+u*(qpre-current[side]);cl=planner.clearance({**current,side:q})
                    if min(cl['arms']['distance'],cl[side+'_bear']['distance'],cl[side+'_table']['distance'],cl[side+'_self']['distance'])<.003:
                        valid=False;break
                if not valid:continue
                q=qpre.copy()
                for u in np.linspace(0,1,41)[1:]:
                    q,e,angle=planner.solve(side,candidate_approach+u*(candidate_target-candidate_approach),orientation,q)
                    cl=planner.clearance({**current,side:q})
                    if e>.003 or angle>.04 or cl['arms']['distance']<.003 or cl[side+'_self']['distance']<.003 or cl[side+'_table']['distance']<.005 or cl[side+'_bear']['distance']<-.001:
                        valid=False;break
                if valid:chosen=(orientation,candidate_target,candidate_approach,entry);break
            if chosen is not None:break
        if chosen is None:raise RuntimeError(side+' has no reachable grasp that passes through the actual aperture')
        orientation,target,approach,entry=chosen;quat=orientation.as_quat()[[3,0,1,2]]
        self.task['arm_extraction'].setdefault('aperture_entries',{})[side]={'center_m':center.tolist(),'entry_m':entry.tolist(),'normal':normal.tolist(),'jaw_center_footprint_inside_opening':True}
        self.grips[side]=np.array([.020,.020])
        self.move({side:approach},2.2,{side:quat},joint=True)
        paw_now=self.paw(side);target+=paw_now-paw
        self.task['arm_extraction'].setdefault('approach_targets',{})[side]=target.tolist()
        self.move({side:target},2.2,{side:quat})
        self.grips[side]=np.array([.012,.012])
        q=array(self.robots[side].get_qpos());q[7:]=self.grips[side]
        self.robots[side].control_dofs_position(q)
        for k in range(120):
            self.step()
            groups,details=self.arm_pad_contacts(side)
            if all(len(g) for g in groups):
                ids=np.unique(np.concatenate(groups));bp=array(self.bear.get_particles_pos());h=self.hands[side]
                local=(bp[ids]-array(h.get_pos()))@matrix(array(h.get_quat()))
                self.bear.fix_particles_to_link(h.idx,particles_idx_local=ids)
                self.arm_bindings[side]={'indices':ids.tolist(),'local_positions':local.tolist()}
                self.task['arm_extraction']['grasps'][side]={'time_s':self.step_count*self.dt,
                    'pad_contacts':details,'attached_indices':ids.tolist(),'actual_tcp_m':self.tcp(side).tolist()}
                break
        else:
            self.task['arm_extraction']['failed_pad_contacts']=details
            raise RuntimeError(side+' paw did not make contact with both gripper pads')
        self.save('grasp_arm_'+side)
        return self.pull_arm(side)

    def pull_arm(self,side):
        quat=array(self.hands[side].get_quat())
        self.start('pull_'+side)
        center,normal,_=self.hole(side)
        # Center the soft paw, whose material centroid is offset from the TCP.
        # Centering only the TCP drags the lower paw against the aperture rim.
        paw_offset=self.tcp(side)-self.paw(side)
        self.move({side:center+normal*.025+paw_offset},2.0,{side:quat})
        self.save('through_hole_'+side)
        center,normal,_=self.hole(side)
        self.move({side:self.tcp(side)+normal*.030},1.6,{side:quat})
        self.move({side:self.tcp(side)+[0,0,.025]},1.2,{side:quat})
        self.wait(.3)
        self.ensure_paw_clearance(side)
        self.record_arm_geometry('after_pull_'+side)
        self.save('arm_out_'+side)

    def ensure_paw_clearance(self,side):
        quat=array(self.hands[side].get_quat())
        for attempt in range(3):
            center,normal,_=self.hole(side)
            points=array(self.bear.get_particles_pos())[self.data['bear'][side+'_paw_indices']]
            signed=(points-center)@normal
            mean=float(signed.mean());minimum=float(signed.min())
            if mean>.035 and minimum>.004:break
            distance=min(.025,max(.005,1.6*max(.035-mean,.005-minimum)))
            self.task['arm_extraction'].setdefault('clearance_adjustments',[]).append({'side':side,'attempt':attempt,'time_s':self.step_count*self.dt,'paw_mean_signed_m':mean,'paw_minimum_signed_m':minimum,'outward_motion_m':distance})
            upward=np.array([0.,0.,.010]);upward-=normal*np.dot(upward,normal)
            self.move({side:self.tcp(side)+normal*distance+upward},1.2,{side:quat});self.wait(.2)

    def save(self,name):
        super().save(name)
        path=self.run/f'state-{name}.npz'
        with np.load(path,allow_pickle=False) as a:state={k:a[k] for k in a.files}
        meta=json.loads(str(state['metadata']));meta['arm_bindings']=self.arm_bindings
        state['metadata']=json.dumps(meta)
        span=slice(self.bear.particle_start,self.bear.particle_end)
        state['bear_free_flags']=self.scene.sim.pbd_solver.particles.free.to_numpy()[span]
        state['bear_attached_link_indices']=self.scene.sim.coupler.particle_attach_info.link_idx.to_numpy()[span]
        np.savez_compressed(path,**state)

    def execute(self):
        endpoint=self.args.stop_after
        if endpoint!='thread_arms':return super().execute()
        continuation=None
        if self.args.resume:
            with np.load(self.args.resume,allow_pickle=False) as state:continuation=json.loads(str(state['metadata']))['phase']
        if continuation in ('regrasp_right_outside','grasp_arm_right','grasp_arm_left','regrasp_left_outside','arm_out_right','thread_arms'):
            self.restore(self.args.resume)
            previous=json.loads((self.args.resume.parent/'report.json').read_text())['wrapping']
            self.task['press_checks']=previous['press_checks']
            if not all(self.task['press_checks'].values()):raise RuntimeError('Aperture continuation requires accepted pressing state')
            self.task['arm_extraction']=previous['arm_extraction']
        else:
            self.args.stop_after='press'
            okay=super().execute()
            self.args.stop_after=endpoint
            if not okay:
                if not getattr(self.args,'debug_extraction',False):return False
                if not (self.args.headless and self.args.no_render and self.args.resume):
                    raise RuntimeError('Diagnostic continuation requires headless/no-render/resume')
                self.task['diagnostic_unaccepted_press_checks']=[k for k,v in self.task['checks'].items() if not v]
                print('DIAGNOSTIC ONLY: continuing to tune extraction from an unaccepted press state',flush=True)
            self.task['press_checks']=self.task['checks'].copy()
            self.record_arm_geometry('after_press')
            self.persist()
        if continuation not in ('arm_out_right','regrasp_left_outside','grasp_arm_left','thread_arms'):self.extract('right',skip_to='aperture' if continuation=='regrasp_right_outside' else 'pull' if continuation=='grasp_arm_right' else None)
        if continuation!='thread_arms':self.extract('left',skip_to='pull' if continuation=='grasp_arm_left' else None)
        # Releasing the second cloth corner changes both aperture planes.
        # Recheck actual paw particles after both arms have been grasped.
        for side in ('right','left'):
            self.start('pull_'+side);self.ensure_paw_clearance(side)
        self.start('pull_left')
        for attempt in range(3):
            low=float(array(self.cloth.get_particles_pos())[:,2].min()-self.table)
            if low>.075:break
            self.task['arm_extraction'].setdefault('hem_height_adjustments',[]).append({'time_s':self.step_count*self.dt,'minimum_cloth_height_above_table_m':low,'upward_motor_motion_m':.012})
            self.move({s:self.tcp(s)+[0,0,.012] for s in self.robots},1.2)
            self.wait(.3)
        self.start('thread_arms');self.wait(.5)
        row=self.record_arm_geometry('final')
        self.save('thread_arms')
        super().evaluate(partial=True)
        checks={**self.task['checks'],
            'accepted_press_state':'diagnostic_unaccepted_press_checks' not in self.task,
            'both_arms_contact_grasped':len(self.arm_bindings)==2,
            'cloth_released_by_both_grippers':not self.bindings,
            'velcro_still_joined':self.seam_ready(),
            'both_paws_outside_holes':all(v['paw_signed_distance_to_hole_m']>.03 for v in row['sides'].values()),
            'whole_paw_particle_groups_outside':all(v['paw_min_signed_distance_to_hole_m']>.003 for v in row['sides'].values()),
            'both_arm_cores_pass_through_holes':all(v['material_curve']['core_passes_through_aperture'] for v in row['sides'].values()),
            'bear_base_still_fixed':self.max_base_drift<=1e-6}
        self.task['checks']=checks
        self.report['status']='passed' if all(checks.values()) else 'failed'
        self.persist()
        return all(checks.values())
