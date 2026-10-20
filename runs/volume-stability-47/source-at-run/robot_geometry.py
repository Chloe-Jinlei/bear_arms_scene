"""Independent FK/IK and coarse rigid clearance checks for dual Franka wrapping."""
from pathlib import Path
import copy,json,math,tomllib,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
HERE=Path(__file__).resolve().parent
ASSET=Path('/home/chloe/Desktop/genesis-local/genesis-world/genesis/assets/xml/franka_emika_panda')
CONFIG=tomllib.loads((HERE/'config.toml').read_text())
HOME=np.array(CONFIG['robots']['home'])
TCP=np.array([0,0,.1029])
DOWN=Rotation.from_quat([1,0,0,0])

def make_model():
    root=ET.parse(ASSET/'panda.xml').getroot()
    root.find('compiler').set('meshdir',str(ASSET/'assets'))
    for name in ('actuator','keyframe','tendon','equality'):
        node=root.find(name)
        if node is not None: root.remove(node)
    world=root.find('worldbody');body=world.find('body');world.remove(body)
    for side in ('left','right'):
        b=copy.deepcopy(body)
        for el in b.iter():
            if 'name' in el.attrib: el.set('name',side+'_'+el.get('name'))
        yaw=CONFIG['robots'][side+'_yaw']
        b.set('pos',' '.join(map(str,CONFIG['robots'][side+'_base'])))
        b.set('quat',f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}')
        world.append(b)
    ET.SubElement(world,'geom',name='table',type='box',pos='0 0 .705',size='.775 .525 .035')
    # These are the procedural teddy's original signed-field primitives. Slight
    # smoothing of its union adds <=2.3 mm; clearance is conservative only above it.
    shapes=[((0,.08,.894),(.140,.110,.181)),((0,.075,1.099),(.117,.096,.105)),
            ((-.102,.083,1.188),(.044,.035,.044)),((.102,.083,1.188),(.044,.035,.044)),
            ((0,-.005,1.087),(.067,.037,.045)),((-.065,.025,.766),(.068,.073,.050)),
            ((.065,.025,.766),(.068,.073,.050))]
    for i,(p,s) in enumerate(shapes):ET.SubElement(world,'geom',name=f'bear_{i}',type='ellipsoid',pos=' '.join(map(str,p)),size=' '.join(map(str,s)))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model)
    return model,data

class Planner:
    def __init__(self):
        self.m,self.d=make_model()
        self.qindices={s:np.array([self.m.joint(f'{s}_joint{i}').qposadr[0] for i in range(1,8)]) for s in ('left','right')}
        self.bodies={s:self.m.body(s+'_hand').id for s in self.qindices}
        self.limits={s:self.m.jnt_range[[self.m.joint(f'{s}_joint{i}').id for i in range(1,8)]] for s in self.qindices}
        self.geoms={s:[i for i in range(self.m.ngeom) if self.m.geom_contype[i] and self.m.body(int(self.m.geom_bodyid[i])).name.startswith(s+'_') and not self.m.body(int(self.m.geom_bodyid[i])).name.endswith('link0')] for s in self.qindices}
        self.bg=[self.m.geom('bear_'+str(i)).id for i in range(7)]
        self.table=self.m.geom('table').id
        self.self_pairs={}
        for s,geoms in self.geoms.items():
            pairs=[]
            for index,i in enumerate(geoms):
                for j in geoms[index+1:]:
                    a,b=(int(self.m.body_weldid[int(self.m.geom_bodyid[g])]) for g in (i,j))
                    if all(self.m.body(int(self.m.geom_bodyid[g])).name.endswith('_finger') for g in (i,j)):continue
                    if a==b or int(self.m.body_weldid[int(self.m.body_parentid[a])])==b or int(self.m.body_weldid[int(self.m.body_parentid[b])])==a:continue
                    pairs.append((i,j))
            self.self_pairs[s]=pairs
        for s in self.qindices:self.d.qpos[self.qindices[s]]=HOME
        for s in self.qindices:
            for i in (1,2):self.d.qpos[self.m.joint(s+'_finger_joint'+str(i)).qposadr[0]]=.001
        mujoco.mj_forward(self.m,self.d)
    def fk(self,s,q):
        self.d.qpos[self.qindices[s]]=q
        mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)
        bid=self.bodies[s];r=self.d.xmat[bid].reshape(3,3)
        return self.d.xpos[bid]+r@TCP,Rotation.from_matrix(r)
    def solve(self,s,p,r,seed):
        def residual(q):
            fp,fr=self.fk(s,q)
            return np.r_[5*(fp-p), (r*fr.inv()).as_rotvec(),.0002*(q-seed)]
        limits=self.limits[s]
        out=least_squares(residual,np.clip(seed,limits[:,0]+1e-6,limits[:,1]-1e-6),bounds=(limits[:,0]+1e-6,limits[:,1]-1e-6),ftol=1e-10,gtol=1e-9,xtol=1e-10,max_nfev=120)
        fp,fr=self.fk(s,out.x)
        error=float(np.linalg.norm(fp-p)); angle=float((r*fr.inv()).magnitude())
        return out.x,error,angle
    def clearance(self,qs):
        for s,q in qs.items():self.d.qpos[self.qindices[s]]=q
        mujoco.mj_forward(self.m,self.d)
        out={}
        for s in qs:
            dis,i,j=min((mujoco.mj_geomDistance(self.m,self.d,i,j,.20,None),i,j) for i,j in self.self_pairs[s])
            out[s+'_self']={'distance':dis,'bodies':[self.m.body(int(self.m.geom_bodyid[k])).name for k in (i,j)]}
            for name,other in [('bear',self.bg),('table',[self.table])]:
                pairs=[(mujoco.mj_geomDistance(self.m,self.d,i,j,.20,None),i,j) for i in self.geoms[s] for j in other]
                dis,i,j=min(pairs)
                out[s+'_'+name]={'distance':dis,'body':self.m.body(int(self.m.geom_bodyid[i])).name,'obstacle':self.m.geom(j).name}
        pairs=[(mujoco.mj_geomDistance(self.m,self.d,i,j,.20,None),i,j) for i in self.geoms['left'] for j in self.geoms['right']]
        dis,i,j=min(pairs);out['arms']={'distance':dis,'bodies':[self.m.body(int(self.m.geom_bodyid[k])).name for k in (i,j)]}
        return out
