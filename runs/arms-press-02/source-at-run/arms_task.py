"""Two physical cloth apertures and contact-gated extraction of soft bear arms."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from wrap_task import WrapExperiment,array,matrix,PHASES

PHASES.update(regrasp_left="左机械臂换抓左侧熊手臂",pull_left="从左侧布孔拉出熊手臂",
              regrasp_right="右机械臂换抓右侧熊手臂",pull_right="从右侧布孔拉出熊手臂",
              thread_arms="双侧手臂已穿出布孔")

class ArmsExperiment(WrapExperiment):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.arm_bindings={}
        self.task['completion_scope']='Wrap, join complementary Velcro strips, then sequentially grasp and draw both soft bear arms through real cloth openings.'
        self.report['scope']=self.task['completion_scope']
        self.task['arm_extraction']={'grasps':{},'measurements':[],
            'arm_topology':'single connected soft tetrahedral bear',
            'grasp_model':'contact-gated local no-slip link constraints; no force closure calibration',
            'cloth_holes':'missing mesh faces, not a transparent texture'}

    def hole(self,side):
        cp=array(self.cloth.get_particles_pos())
        ids=self.data['cotton'][side+'_hole_indices']
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
            row['sides'][s]={'hole_center_m':c.tolist(),'hole_normal':n.tolist(),'paw_center_m':paw.tolist(),
                'paw_signed_distance_to_hole_m':float(np.dot(paw-c,n)),
                'hole_min_boundary_radius_m':float(np.linalg.norm(p-c,axis=1).min())}
        self.task['arm_extraction']['measurements'].append(row)
        return row

    def record(self,bp,cp):
        super().record(bp,cp)
        if self.phase.startswith(('regrasp','pull')) or self.phase=='thread_arms':
            row=self.record_arm_geometry('trajectory')
            self.log.write(json.dumps({'arm_geometry':row})+'\n');self.log.flush()

    def step(self):
        super().step()
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
        self.move({side:center+[sign*.09,0,0]},1.8,{side:quat})
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
        self.move({side:center+[sign*.025,0,0]},1.8,{side:quat})
        self.save('through_hole_'+side)
        center,normal,_=self.hole(side)
        self.move({side:center+[sign*.080,0,.010]},2.2,{side:quat})
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
        self.extract('left')
        self.extract('right')
        self.start('thread_arms');self.wait(.5)
        row=self.record_arm_geometry('final')
        self.save('thread_arms')
        checks={**self.task['press_checks'],
            'both_arms_contact_grasped':len(self.arm_bindings)==2,
            'cloth_released_by_both_grippers':not self.bindings,
            'velcro_still_joined':self.velcro.metrics()['active_bonds']>=6,
            'both_paws_outside_holes':all(v['paw_signed_distance_to_hole_m']>.03 for v in row['sides'].values()),
            'bear_base_still_fixed':self.max_base_drift<=1e-6}
        self.task['checks']=checks
        self.report['status']='passed' if all(checks.values()) else 'failed'
        self.persist()
        return all(checks.values())
