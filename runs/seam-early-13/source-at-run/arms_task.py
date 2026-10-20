"""Two physical cloth apertures and contact-gated extraction of soft bear arms."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from wrap_task import WrapExperiment,array,matrix,PHASES
from arm_geometry import ArmMaterialCurve,find_hole_boundary,aperture_measurement

PHASES.update(regrasp_left="左机械臂换抓左侧熊手臂",pull_left="从左侧布孔拉出熊手臂",
              regrasp_right="右机械臂换抓右侧熊手臂",pull_right="从右侧布孔拉出熊手臂",
              thread_arms="双侧手臂已穿出布孔")

class ArmsExperiment(WrapExperiment):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.arm_bindings={}
        from robot_geometry import Planner
        self.clearance_planner=Planner()
        self.arm_curves={s:ArmMaterialCurve(self.data['bear']['points'],self.data['bear']['tetra'],s,
            [*self.config['bear']['position_xy'],self.table+self.config['bear']['base_clearance']]) for s in self.robots}
        material_points=self.data['cotton']['points'].copy()
        material_points[:,:2]-=self.config['cloth']['position_xy']
        self.hole_boundaries={s:find_hole_boundary(material_points,self.data['cotton']['faces'],
            self.report['geometry']['cotton']['holes'][s]['center_material_xy_m']) for s in self.robots}
        self.task['completion_scope']='Wrap, join complementary Velcro strips, then sequentially grasp and draw both soft bear arms through real cloth openings.'
        self.report['scope']=self.task['completion_scope']
        self.task['arm_extraction']={'grasps':{},'measurements':[],
            'minimum_joined_patch':{'bonds':3,'span_m':.025,'source_patch_area_m2':.0004,
                'scope':'Geometric distributed attachment criterion, not a calibrated load or peel rating.'},
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
        initial_tip=self.tcp('left').copy()
        for attempt in range(16):
            if self.seam_ready():return
            cp=array(self.cloth.get_particles_pos());g=self.velcro.geometry.candidates(cp)
            bonded=[b['source_node'] for b in self.velcro.metrics()['activation_events'] if b['active']]
            valid=(g['normal_dot']<0)&(g['source_facing_gap_m']>=-.001)&(g['target_facing_gap_m']>=-.001)&~np.isin(g['source_nodes'],bonded)
            if not valid.any():break
            index=np.flatnonzero(valid)[np.argmin(g['distance_m'][valid])]
            print(f"Seam refine {attempt}: gap={g['distance_m'][index]:.4f} m, normal dot={g['normal_dot'][index]:.3f}, bonds={len(bonded)}",flush=True)
            current=Rotation.from_quat(array(self.hands['left'].get_quat())[[1,2,3,0]])
            if g['normal_dot'][index]>-.88:
                source=g['target_active_normals'][index];desired=-g['source_active_normals'][index]
                axis=np.cross(source,desired);axis/=max(np.linalg.norm(axis),1e-12)
                angle=min(.18,float(np.arccos(np.clip(np.dot(source,desired),-1,1))))
                right_rotation=Rotation.from_quat(array(self.hands['right'].get_quat())[[1,2,3,0]])
                rotation=Rotation.from_rotvec(axis*angle)*right_rotation
                if not self.press_pose_clear('right',self.tcp('right'),rotation):break
                target_quat=rotation.as_quat()[[3,0,1,2]]
                self.move({'right':self.tcp('right')},.55,{'right':target_quat});self.wait(.15)
                self.task['seam_alignment'].append({'normal_alignment_attempt':attempt,'normal_dot':float(g['normal_dot'][index]),'time_s':self.step_count*self.dt})
                self.save('press_normal_'+str(attempt))
                continue
            target=(cp[g['target_nodes'][index]]*g['barycentric'][index,:,None]).sum(0)
            delta=target-cp[g['source_nodes'][index]];distance=np.linalg.norm(delta)
            if distance<.0035:
                self.wait(.5)
                continue
            delta*=min(.004,max(0.,distance-.003))/distance
            destination=self.tcp('left')+delta
            if np.linalg.norm(destination-initial_tip)>.04:break
            if not self.press_pose_clear('left',destination,current):break
            lifted=current.as_quat()[[3,0,1,2]]
            self.move({'left':destination},.55,{'left':lifted});self.wait(.15)
            self.task['seam_alignment'].append({'surface_guided_attempt':attempt,'time_s':self.step_count*self.dt,
                'measured_gap_m':float(distance),'motor_tcp_shift_m':delta.tolist(),'metrics':self.velcro.metrics()})
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
        return m['active_bonds']>=3 and m['bonded_width_span_m']>=.025-1e-8 and m['bonded_hook_vertex_area_m2']>=.0004

    def restore(self,path):
        stage=super().restore(path)
        if stage=='press_separate':
            self.start('press');self.refine_seam_current_pose()
            if self.seam_ready():
                self.save('press');return 'press'
        if stage=='press':
            self.start('press');self.tighten_seam();self.save('press')
        return stage

    def release_cloth(self,side):
        binding=self.bindings.pop(side)
        self.cloth.release_particle(particles_idx_local=binding['indices'])
        self.task['arm_extraction'].setdefault('cloth_release_events',[]).append(
            {'side':side,'time_s':self.step_count*self.dt,'indices':binding['indices'],
             'active_bonds':self.velcro.metrics()['active_bonds']})
        # Clear the seam with a narrow jaw opening before opening fully outside.
        self.grips[side]=np.array([.006,.006])
        q=array(self.robots[side].get_qpos());q[7:]=self.grips[side]
        self.robots[side].control_dofs_position(q)

    def extract(self,side):
        sign=-1 if side=='left' else 1
        self.start('regrasp_'+side)
        self.release_cloth(side)
        self.move({side:self.tcp(side)+[sign*.035,.075,.035]},1.5)
        # Palm remains outside the teddy, with fingertip direction facing inward.
        rot=Rotation.from_matrix(np.array([[0.,0.,-sign],[0.,1.,0.],[sign,0.,0.]]))
        quat=rot.as_quat()[[3,0,1,2]]
        self.grips[side]=np.array([.035,.035])
        self.move({side:[sign*.31,.19,1.045]},2.0,{side:quat})
        self.save('regrasp_'+side+'_outside')
        self.record_arm_geometry('before_grasp_'+side)
        center,normal,_=self.hole(side)
        paw=self.paw(side)
        target=paw+np.array([sign*.008,0,.012])
        direction=target-center
        direction/=np.linalg.norm(direction)
        jaw=np.array([0.,0.,1.]);jaw-=direction*np.dot(jaw,direction);jaw/=np.linalg.norm(jaw)
        xaxis=np.cross(jaw,direction)
        quat=Rotation.from_matrix(np.column_stack([xaxis,jaw,direction])).as_quat()[[3,0,1,2]]
        self.grips[side]=np.array([.028,.028])
        self.move({side:center-direction*.085},1.8,{side:quat})
        # Approach the exposed upper paw from the actual aperture, recalculating
        # the contact target from simulated body vertices after each motion.
        paw=self.paw(side)
        target=paw+np.array([sign*.008,0,.012])
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
        self.start('pull_'+side)
        center,normal,_=self.hole(side)
        self.move({side:center+normal*.025},1.8,{side:quat})
        self.save('through_hole_'+side)
        center,normal,_=self.hole(side)
        self.move({side:center+normal*.080+[0,0,.010]},2.2,{side:quat})
        self.wait(.3)
        self.record_arm_geometry('after_pull_'+side)
        self.save('arm_out_'+side)

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
        self.args.stop_after='press'
        okay=super().execute()
        self.args.stop_after=endpoint
        if not okay:return False
        self.task['press_checks']=self.task['checks'].copy()
        self.record_arm_geometry('after_press')
        self.persist()
        self.extract('right')
        self.extract('left')
        self.start('thread_arms');self.wait(.5)
        row=self.record_arm_geometry('final')
        self.save('thread_arms')
        super().evaluate(partial=True)
        checks={**self.task['checks'],
            'both_arms_contact_grasped':len(self.arm_bindings)==2,
            'cloth_released_by_both_grippers':not self.bindings,
            'velcro_still_joined':self.seam_ready(),
            'both_paws_outside_holes':all(v['paw_signed_distance_to_hole_m']>.03 for v in row['sides'].values()),
            'both_arm_cores_pass_through_holes':all(v['material_curve']['core_passes_through_aperture'] for v in row['sides'].values()),
            'bear_base_still_fixed':self.max_base_drift<=1e-6}
        self.task['checks']=checks
        self.report['status']='passed' if all(checks.values()) else 'failed'
        self.persist()
        return all(checks.values())
