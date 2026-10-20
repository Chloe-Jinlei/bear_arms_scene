from pathlib import Path
import sys,json,hashlib,os
import numpy as np
import trimesh,tetgen,fast_simplification
from skimage.measure import marching_cubes
DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(DEST))
from asset_primitives import bear_field,tetra_boundary,cylindrical_uv,write_obj
from arm_geometry import find_hole_boundary
BASE=DEST/'parent-assets'

def field(p):
    result=bear_field(p)
    for sign in (-1,1):
        centers=np.array([[.094,0,.29],[.133,0,.275],[.155,0,.25],[.155,-.005,.22]])
        centers[:,0]*=sign
        radii=np.array([.027,.025,.021,.023])
        for a,b,ra,rb in zip(centers[:-1],centers[1:],radii[:-1],radii[1:]):
            scale=np.array([1/.6,1.,1.]);axis=(b-a)*scale
            t=np.clip(np.sum((p-a)*scale*axis,axis=-1)/np.dot(axis,axis),0,1)
            d=(np.linalg.norm((p-(a+t[...,None]*(b-a)))*scale,axis=-1)-(ra+t*(rb-ra)))*.6
            k=.004;h=np.maximum(k-np.abs(result-d),0)/k
            result=np.minimum(result,d)-h*h*k*.25
    return result

spacing=.003
low=np.array([-.205,-.16,-.012]);high=np.array([.205,.14,.510])
axes=[np.arange(a,b+spacing/2,spacing) for a,b in zip(low,high)]
grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
v,f,_,_=marching_cubes(field(grid),level=0,spacing=(spacing,)*3)
v+=low;v[:,2]=np.maximum(v[:,2],0);v*=.48/v[:,2].max()
mesh=trimesh.Trimesh(v,f,process=True);trimesh.repair.fix_normals(mesh)
assert mesh.is_watertight and len(mesh.split())==1
pv,pf=fast_simplification.simplify(mesh.vertices,mesh.faces,target_count=2200,agg=5.)
pv[:,2]=np.maximum(pv[:,2],0)
pv[pv[:,2]<.0001,2]=0
physical=trimesh.Trimesh(pv,pf,process=True);trimesh.repair.fix_normals(physical)
bottom=physical.vertices[:,2]<.0001
trimesh.smoothing.filter_taubin(physical,lamb=.5,nu=.53,iterations=10)
physical.vertices[bottom,2]=0
physical.vertices[:,2]=np.maximum(physical.vertices[:,2],0)
assert physical.is_watertight
tet=tetgen.TetGen(physical.vertices,physical.faces.astype(np.int32))
points,tetra,_,_=tet.tetrahedralize(quality=True,minratio=1.5,mindihedral=5,nobisect=True,
    maxvolume=5e-6,fixedvolume=True,steinerleft=1800,quiet=True)
vol=np.linalg.det(points[tetra[:,1:]]-points[tetra[:,:1]])/6
tetra[vol<0]=tetra[vol<0][:,[0,2,1,3]]
if np.abs(vol).min()<=1e-14:
    print(points[tetra[np.argsort(np.abs(vol))[:3]]],flush=True)
assert np.abs(vol).min()>1e-14, (np.abs(vol).min(),np.count_nonzero(np.abs(vol)<1e-14))
used,sf=tetra_boundary(points,tetra);shell=trimesh.Trimesh(points[used],sf,process=False)
assert shell.is_watertight and len(shell.split())==1
vv,vf=fast_simplification.simplify(mesh.vertices,mesh.faces,target_count=12000,agg=5.)
uv=cylindrical_uv(vv)
AS=DEST/'assets'
write_obj(AS/'bear.obj',vv,vf,uv,'bear_fur.png',seam=True)
arm_groups={}
for side,sign in [('left',-1),('right',1)]:
    arm_groups[side+'_arm_indices']=np.flatnonzero((sign*points[:,0]>.14)&(points[:,2]>.145)&(points[:,2]<.285))
    arm_groups[side+'_paw_indices']=np.flatnonzero((sign*points[:,0]>.14)&(points[:,2]<.243)&(points[:,2]>.19))
np.savez_compressed(AS/'bear.npz',points=points,tetra=tetra,surface_points=points[used],
    surface_faces=sf,surface_point_indices=used,visual_points=vv,visual_faces=vf,uv=uv,
    fixed_indices=np.flatnonzero(points[:,2]<=.012),**arm_groups)

# Cut real mesh openings while preserving every remaining material coordinate.
# Remove triangles intersecting a circular template, then remove orphan nodes.
old=np.load(BASE/'cotton.npz');cp=old['points'];cf=old['faces'];keep=np.ones(len(cf),bool)
holes={'left':[-.22,-.05],'right':[.22,-.015]}
radius=.029
for c in holes.values():
    # Conservative per-triangle clipping on the original 12.5 mm grid.
    center=np.r_[c,0.]
    closest=trimesh.triangles.closest_point(cp[cf],np.tile(center,(len(cf),1)))
    keep &= np.linalg.norm(closest-center,axis=1)>radius
usedc,newfaces=np.unique(cf[keep],return_inverse=True);newfaces=newfaces.reshape(-1,3)
cp=cp[usedc];uvc=old['uv'][usedc]
edges=np.sort(np.concatenate([newfaces[:,[0,1]],newfaces[:,[1,2]],newfaces[:,[2,0]]]),axis=1)
ue,cnt=np.unique(edges,axis=0,return_counts=True);be=ue[cnt==1]
holedata={};holemeta={}
for side,c in holes.items():
    ids=find_hole_boundary(cp,newfaces,c)
    he=be[np.isin(be[:,0],ids)&np.isin(be[:,1],ids)]
    assert len(ids)>8
    holedata[side+'_hole_indices']=ids;holedata[side+'_hole_edges']=he
    holemeta[side]={'center_material_xy_m':c,'boundary_indices':ids.tolist(),
        'boundary_edges':he.tolist(),'bounding_size_m':np.ptp(cp[ids,:2],axis=0).tolist()}
write_obj(AS/'cotton.obj',cp,newfaces,uvc,'cotton_weave.png')
np.savez_compressed(AS/'cotton.npz',points=cp,faces=newfaces,uv=uvc,
    left_edge_indices=np.flatnonzero(np.isclose(cp[:,0],-.43)),
    right_edge_indices=np.flatnonzero(np.isclose(cp[:,0],.43)),**holedata)
meta=json.loads((BASE/'metadata.json').read_text())
meta['bear'].update(height_m=float(np.ptp(points[:,2])),bounds_m=[points.min(0).tolist(),points.max(0).tolist()],
    physics_vertices=len(points),tetrahedra=len(tetra),surface_vertices=len(used),surface_faces=len(sf),
    visual_vertices=len(vv),visual_faces=len(vf),volume_m3=float(np.abs(vol).sum()),
    features=['two soft connected hanging arms','flat fixed base','round ears','muzzle','two small feet'],
    arm_topology='single connected tetrahedral soft body including shoulders and both arms',
    arm_centerline_centers_local_m=[[.094,0,.29],[.133,0,.275],[.155,0,.25],[.155,-.005,.22]],
    arm_cross_section_scale_xyz=[.6,1.,1.],
    paw_indices={s:arm_groups[s+'_paw_indices'].tolist() for s in holes})
area=trimesh.Trimesh(cp,newfaces,process=False).area
meta['cotton'].update(vertices=len(cp),faces=len(newfaces),area_m2=float(area),suggested_mass_kg=float(area*.18),
    holes=holemeta,hole_topology='actual missing triangles; three boundary loops; no faces span openings',
    original_vertex_indices=usedc.tolist())
meta['parent_assets_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.is_file()}
meta['provenance']='Original local procedural extension of preserved Velcro scene; torso field, cloth extents and strip textures retained.'
(AS/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'bear_vertices':len(points),'tets':len(tetra),'min_tet_volume':float(np.abs(vol).min()),'holes':holemeta,'cloth_area':area},indent=2))
