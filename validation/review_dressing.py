"""Independent saved-state checks for the complete physical dressing action."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import trimesh

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
run=a.run.resolve();source=run/'source-at-run';sys.path.insert(0,str(source))
from arm_geometry import ArmMaterialCurve,find_hole_boundary,aperture_measurement
from wrap_contact import ContactGeometry
from velcro_contact import VelcroGeometry
report=json.loads((run/'report.json').read_text());config=report['config']
b=np.load(source/'assets/bear.npz');c=np.load(source/'assets/cotton.npz')
offset=np.r_[config['bear']['position_xy'],config['table']['height']+config['bear']['base_clearance']]
rest=b['points']+offset;tet=b['tetra'];faces=c['faces']
geometry=ContactGeometry(rest,tet,c['points'],faces)
seam=VelcroGeometry(c['points'],faces,config)
boundaries={s:find_hole_boundary(c['points'],faces,report['geometry']['cotton']['holes'][s]['center_material_xy_m']) for s in ('left','right')}
curves={s:ArmMaterialCurve(rest,tet,s,offset,report['geometry']['bear'].get('arm_centerline_centers_local_m')) for s in boundaries}
records=[];states={};base=np.array(report['simulation']['fixed_particle_indices'])
edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
length=np.linalg.norm(c['points'][edges[:,0]]-c['points'][edges[:,1]],axis=1)
rest_volume=np.linalg.det(rest[tet[:,1:]]-rest[tet[:,:1]])/6
for p in run.rglob('state-*.npz'):
    with np.load(p,allow_pickle=False) as archive:state={k:archive[k].copy() for k in archive.files}
    meta=json.loads(str(state['metadata']));bp=state['bear_positions'];cp=state['cloth_positions'];label=p.stem[6:]
    if label not in states or meta['step']>=states[label][1]['step']:states[label]=(state,meta)
    volume=np.linalg.det(bp[tet[:,1:]]-bp[tet[:,:1]])/6
    item={'stage':label,'step':meta['step'],'finite':bool(np.isfinite(bp).all() and np.isfinite(cp).all()),
        'fixed_base_max_drift_m':float(np.linalg.norm(bp[base]-rest[base],axis=1).max()),
        'head_horizontal_displacement_m':float(np.linalg.norm((bp[b['points'][:,2]>.32]-rest[b['points'][:,2]>.32]).mean(0)[:2])),
        'cloth_max_stretch_ratio':float((np.linalg.norm(cp[edges[:,0]]-cp[edges[:,1]],axis=1)/length).max()),
        'inverted_tets':int(np.sum(volume<0)),
        'inverted_volume_fraction':float(np.abs(volume[volume<0]).sum()/np.abs(rest_volume).sum()),
        'cloth_body_penetration':geometry.validate(bp,cp),
        'arm_apertures':{s:aperture_measurement(curves[s].positions(bp),cp[boundaries[s]],s) for s in boundaries}}
    records.append(item)
records.sort(key=lambda x:x['step'])
checks={
    'reported_complete':report['status']=='passed',
    'fresh_simulation_not_resumed':'initialization' not in report['wrapping'],
    'finite_states':all(x['finite'] for x in records),
    'base_fixed_all_saved_stages':all(x['fixed_base_max_drift_m']<1e-6 for x in records),
    'no_deep_cloth_body_penetration_saved_stages':all(x['cloth_body_penetration']['max_penetration_m']<.003 for x in records),
    'no_substantial_negative_body_volume':all(x['inverted_volume_fraction']<1e-4 for x in records),
    'two_closed_real_apertures':len(boundaries)==2,
    'connected_cloth':len(trimesh.Trimesh(c['points'],faces,process=False).split(only_watertight=False))==1,
    'connected_bear_surface':len(geometry.mesh(rest).split())==1,
}
final,meta=states['thread_arms'];press,pressmeta=states['press']
final_row=max((x for x in records if x['stage']=='thread_arms'),key=lambda x:x['step'])
press_row=max((x for x in records if x['stage']=='press'),key=lambda x:x['step'])
checks.update(

    both_arm_cores_pass_through_openings=all(v['core_passes_through_aperture'] and v['tip_signed_distance_m']>.025 for v in final_row['arm_apertures'].values()),
    cloth_not_overstretched_final=final_row['cloth_max_stretch_ratio']<1.3,
    bear_head_remains_over_fixed_base=final_row['head_horizontal_displacement_m']<.05,
    cloth_free_of_robot_attachments=bool(np.all(final['cloth_attached_link_indices']<0)),
    cloth_free_of_world_pins=bool(np.all(final['cloth_free_flags'])),
    both_arms_attached_to_grippers=all(len(meta['arm_bindings'].get(s,{}).get('indices',[]))>0 for s in boundaries),
)
events=meta['velcro']['bonds'];active=[e for e in events if e['active']]
nodes=np.array([e['source_node'] for e in active],int)
checks['seam_has_actual_active_contact']=len(active)>=1
press_paw_clearances={}
for side,ids in boundaries.items():
    points=press['cloth_positions'][ids];center=points.mean(0);_,_,basis=np.linalg.svd(points-center);normal=basis[-1]
    if normal[0]*(-1 if side=='left' else 1)<0:normal=-normal
    distances=(press['bear_positions'][b[side+'_paw_indices']]-center)@normal
    press_paw_clearances[side]={'minimum_signed_distance_m':float(distances.min()),'mean_signed_distance_m':float(distances.mean())}
checks['both_paws_not_fully_exposed_before_extraction']=all(v['minimum_signed_distance_m']<0 for v in press_paw_clearances.values())
paw_clearances={}
for side,ids in boundaries.items():
    points=final['cloth_positions'][ids];center=points.mean(0);_,_,basis=np.linalg.svd(points-center);normal=basis[-1]
    if normal[0]*(-1 if side=='left' else 1)<0:normal=-normal
    distances=(final['bear_positions'][b[side+'_paw_indices']]-center)@normal
    paw_clearances[side]={'minimum_signed_distance_m':float(distances.min()),'mean_signed_distance_m':float(distances.mean())}
checks['whole_paw_particle_groups_outside_openings']=all(v['minimum_signed_distance_m']>.003 for v in paw_clearances.values())

errors=seam.errors(final['cloth_positions'],active)
checks['adhesive_errors_below_break_threshold']=bool(len(errors) and np.max(errors)<config['velcro']['break_error_m'])
checks['all_adhesion_activated_by_actual_contact']=bool(events) and all(e['activation_distance_m']<=.005+1e-7 and e['dwell_s']>=.08-1e-7 and e['activation_normal_dot']<=-np.cos(np.deg2rad(35))+1e-7 and min(e['source_facing_gap_m'],e['target_facing_gap_m'])>=-.001-1e-7 for e in events)
gr=report['wrapping']['arm_extraction']['grasps']
checks['both_arm_grasps_after_adhesion']=len(gr)==2 and all(g['time_s']>=pressmeta['step']*report['simulation']['step_dt_s'] for g in gr.values())
checks['both_arm_grasps_gated_by_two_pads']=len(gr)==2 and all(len(g['pad_contacts'])==2 and all(p['indices'] and max(p['distance_m'])<=.005+1e-7 for p in g['pad_contacts']) for g in gr.values())
checks['seam_never_completely_detached_during_extraction']=report['wrapping']['arm_extraction']['minimum_active_bonds']>0
hash_checks=[]
for key,folder in [('implementation_sha256',source),('source_sha256',source/'assets')]:
    for name,h in report[key].items():hash_checks.append(hashlib.sha256((folder/name).read_bytes()).hexdigest()==h)
checks['exact_run_source_and_assets_preserved']=all(hash_checks)
result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'stage_measurements':records,
    'active_bonds_final':len(active),'whole_paw_clearances':paw_clearances,'press_paw_clearances':press_paw_clearances,'maximum_adhesive_error_m':float(max(errors,default=0.)),
    'limitations':['Discrete sampled contact checks at saved stages, not continuous collision detection.',
        'Soft-body negative tetrahedra are quantified, not concealed; small residual inversion is not a high-fidelity FEM result.',
        'Grasps and Velcro use ideal geometric constraints; one or more actual-contact point attachments do not certify full-width adhesive coverage, calibrated pulling force, or peel strength.',
        'Final grippers hold the extracted arms.',
        'Additional cloth/body sampled contact uses the deforming body as a one-way obstacle; native PBD and robot/body coupling remain active.']}
(run/'independent-dressing-review.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':result['status'],'checks':checks,'active_bonds_final':len(active)},indent=2))
sys.exit(0 if all(checks.values()) else 1)
