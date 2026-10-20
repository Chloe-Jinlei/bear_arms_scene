"""Material-point and aperture measurements; never writes simulation state."""
import numpy as np

def find_hole_boundary(points,faces,center):
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    edges,count=np.unique(edges,axis=0,return_counts=True);edges=edges[count==1]
    adjacency={}
    for a,b in edges:
        adjacency.setdefault(int(a),[]).append(int(b));adjacency.setdefault(int(b),[]).append(int(a))
    remaining=set(adjacency);loops=[]
    while remaining:
        component=set();stack=[next(iter(remaining))]
        while stack:
            v=stack.pop()
            if v in component:continue
            component.add(v);stack.extend(adjacency[v])
        remaining-=component
        loop=ordered_boundary(edges[np.isin(edges[:,0],list(component))])
        if np.ptp(points[loop,0])<.2:loops.append(loop)
    assert len(loops)==2,'Exactly two true closed interior mesh boundaries are required'
    return min(loops,key=lambda ids:np.linalg.norm(points[ids,:2].mean(0)-center))

def ordered_boundary(edges):
    adjacency={}
    for a,b in np.asarray(edges):
        adjacency.setdefault(int(a),[]).append(int(b));adjacency.setdefault(int(b),[]).append(int(a))
    assert all(len(n)==2 for n in adjacency.values()),'Aperture boundary must be a simple loop'
    first=min(adjacency);order=[first];previous=None;current=first
    while True:
        following=next(x for x in adjacency[current] if x!=previous)
        if following==first:break
        assert following not in order
        order.append(following);previous,current=current,following
    assert len(order)==len(adjacency)
    return np.asarray(order)

class ArmMaterialCurve:
    def __init__(self,points,tetra,side,offset,centers=None):
        sign=-1 if side=='left' else 1
        centers=np.array(centers if centers is not None else [[.094,0,.29],[.143,0,.275],[.166,0,.25],[.165,-.005,.22]],dtype=float)
        centers[:,0]*=sign;centers+=offset
        curve=np.vstack([a+(b-a)*np.linspace(0,1,9,endpoint=False)[:,None] for a,b in zip(centers[:-1],centers[1:])]+[centers[-1:]])
        triangles=np.asarray(points)[tetra];basis=np.transpose(triangles[:,1:]-triangles[:,:1],(0,2,1))
        inverse=np.linalg.inv(basis)
        nodes=[];weights=[]
        for p in curve:
            w=np.einsum('tij,tj->ti',inverse,p-triangles[:,0]);w=np.c_[1-w.sum(1),w]
            index=np.argmax(w.min(1));assert w[index].min()>-1e-4,(side,p,w[index])
            nodes.append(tetra[index]);weights.append(w[index])
        self.nodes=np.asarray(nodes);self.weights=np.asarray(weights)

    def positions(self,body):
        return (np.asarray(body)[self.nodes]*self.weights[...,None]).sum(1)

def point_in_polygon(p,polygon):
    a=polygon;b=np.roll(a,-1,axis=0)
    crossing=((a[:,1]>p[1])!=(b[:,1]>p[1])) & (p[0]<(b[:,0]-a[:,0])*(p[1]-a[:,1])/(b[:,1]-a[:,1]+1e-30)+a[:,0])
    edge=b-a;t=np.clip(np.einsum('ij,ij->i',p-a,edge)/np.maximum(np.sum(edge*edge,axis=1),1e-30),0,1)
    clearance=float(np.linalg.norm(p-a-t[:,None]*edge,axis=1).min())
    return bool(crossing.sum()%2),clearance

def aperture_measurement(curve,boundary,side):
    center=boundary.mean(0);_,_,basis=np.linalg.svd(boundary-center)
    normal=basis[-1];sign=-1 if side=='left' else 1
    if normal[0]*sign<0:normal=-normal
    signed=(curve-center)@normal;polygon=(boundary-center)@basis[:2].T
    crossings=[]
    for i in range(len(curve)-1):
        if signed[i]*signed[i+1]<=0 and abs(signed[i+1]-signed[i])>1e-9:
            t=-signed[i]/(signed[i+1]-signed[i]);p=curve[i]+t*(curve[i+1]-curve[i])
            inside,clearance=point_in_polygon((p-center)@basis[:2].T,polygon)
            crossings.append({'point_m':p.tolist(),'inside_aperture':inside,'boundary_clearance_m':clearance})
    return {'root_signed_distance_m':float(signed[0]),'tip_signed_distance_m':float(signed[-1]),
            'core_crossings':crossings,'core_passes_through_aperture':any(x['inside_aperture'] and x['boundary_clearance_m']>.002 for x in crossings),
            'hole_planarity_error_m':float(np.abs((boundary-center)@normal).max())}
