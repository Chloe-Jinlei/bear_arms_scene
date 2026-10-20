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
        # Development checkpoint: extraction is implemented after validating
        # this new mesh's full approach/wrap/contact behavior.
        raise RuntimeError('Arm extraction controller development checkpoint reached')
