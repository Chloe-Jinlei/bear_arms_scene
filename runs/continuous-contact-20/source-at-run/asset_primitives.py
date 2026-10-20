"""Generate the original armless teddy and cotton rectangle, in metres.

Run with the existing Genesis environment's Python. No downloaded assets or
changes to the installed Genesis package are required. Bear local base is z=0,
front is -Y, and the rectangle lies in XY, centred at the origin.
"""
from __future__ import annotations

import json
from pathlib import Path

import fast_simplification
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import tetgen
import trimesh

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def bear_field(p):
    """Smooth union of a torso, head, ears, muzzle and two small seated feet."""
    shapes = [
        ((0, 0, .151), (.140, .110, .181)),
        ((0, -.005, .356), (.117, .096, .105)),
        ((-.102, .003, .445), (.044, .035, .044)),
        ((.102, .003, .445), (.044, .035, .044)),
        ((0, -.085, .344), (.067, .037, .045)),
        ((-.065, -.055, .023), (.068, .073, .050)),
        ((.065, -.055, .023), (.068, .073, .050)),
    ]
    fields = [(np.linalg.norm((p - c) / r, axis=-1) - 1) * min(r)
              for c, r in shapes]
    result = fields[0]
    for other in fields[1:]:
        k = .009
        h = np.maximum(k - np.abs(result - other), 0) / k
        result = np.minimum(result, other) - h * h * k * .25
    return np.maximum(result, -p[..., 2])


def tetra_boundary(points, tetra):
    faces = np.concatenate([tetra[:, [1, 2, 3]], tetra[:, [0, 3, 2]],
                            tetra[:, [0, 1, 3]], tetra[:, [0, 2, 1]]])
    opposite = np.concatenate([tetra[:, 0], tetra[:, 1], tetra[:, 2], tetra[:, 3]])
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    assert counts.max() == 2
    keep = counts[inverse] == 1
    faces, opposite = faces[keep], opposite[keep]
    tri = points[faces]
    inward = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                       points[opposite] - tri[:, 0]) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    used, remapped = np.unique(faces, return_inverse=True)
    return used, remapped.reshape(-1, 3)


def cylindrical_uv(points, height=.48):
    return np.column_stack([.5 + np.arctan2(points[:, 0], -points[:, 1]) / (2 * np.pi),
                            points[:, 2] / height])


def write_obj(path, vertices, faces, uv=None, texture=None, seam=False):
    """UV seam uses independent vt indices; geometric vertices stay connected."""
    if uv is not None:
        texture_uv = np.asarray(uv).copy()
        uv_faces = faces.copy()
        if seam:
            corner_uv = uv[faces].copy()
            crossed = np.ptp(corner_uv[..., 0], axis=1) > .5
            extra = {}
            for i in np.flatnonzero(crossed):
                for corner in np.flatnonzero(corner_uv[i, :, 0] < .5):
                    vertex = int(faces[i, corner])
                    if vertex not in extra:
                        extra[vertex] = len(uv) + len(extra)
                    uv_faces[i, corner] = extra[vertex]
            if extra:
                texture_uv = np.concatenate([uv, uv[list(extra)] + [1, 0]])
    with path.open("w") as f:
        f.write("# Original procedural bear-wrap asset. Units: metres.\n")
        if texture:
            f.write(f"mtllib {path.stem}.mtl\nusemtl fabric\n")
        for v in vertices:
            f.write("v " + " ".join(f"{x:.10g}" for x in v) + "\n")
        if uv is not None:
            for coord in texture_uv:
                f.write("vt " + " ".join(f"{x:.10g}" for x in coord) + "\n")
        for i, face in enumerate(faces + 1):
            entries = [f"{j}/{uv_faces[i,k]+1}" if uv is not None else str(j)
                       for k, j in enumerate(face)]
            f.write("f " + " ".join(entries) + "\n")
    if texture:
        path.with_suffix(".mtl").write_text(
            f"newmtl fabric\nKd 1.0 1.0 1.0\nKa 0.2 0.2 0.2\nKs 0.02 0.02 0.02\nNs 5\nmap_Kd {texture}\n")


def make_bear_texture():
    n = 2048
    rng = np.random.default_rng(41)
    noise = rng.normal(0, 1, (n, n))
    grain = gaussian_filter(noise, .7) * 4 + gaussian_filter(noise, 12) * 20
    yy, xx = np.mgrid[:n, :n]
    base = np.array([155, 105, 66], float)
    glow = 9 * np.cos((xx / n - .5) * np.pi * 2)
    rgb = np.clip(base + (grain + glow)[..., None], 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    # Map desired angle coordinate and physical height to image pixels.
    def box(u, z, du, dz):
        return ((u-du)*n, (1-(z+dz)/.48)*n, (u+du)*n, (1-(z-dz)/.48)*n)
    def line(points, fill, width):
        draw.line([(u*n, (1-z/.48)*n) for u, z in points], fill=fill, width=width, joint="curve")
    cream = (224, 194, 145)
    brown = (70, 43, 29)
    # Embroidered belly, muzzle and ear insets.
    draw.ellipse(box(.5, .180, .142, .107), fill=cream)
    draw.ellipse(box(.5, .340, .112, .041), fill=(232, 207, 162))
    for u in (.306, .694):
        draw.ellipse(box(u, .439, .050, .024), fill=(205, 156, 112))
        draw.ellipse(box(u, .440, .034, .017), fill=(220, 178, 134))
    # Glossy embroidered eyes; mapping keeps both eyes on the front of the head.
    for u in (.437, .563):
        draw.ellipse(box(u, .387, .0148, .0107), fill=(37, 26, 22))
        draw.ellipse(box(u-.004, .390, .0039, .0030), fill=(248, 236, 211))
        draw.ellipse(box(u+.004, .3825, .0021, .0015), fill=(120, 85, 60))
    # Rounded triangular nose, stitched centreline and friendly curved smile.
    nose = [(.481, .355), (.519, .355), (.520, .349), (.508, .341),
            (.500, .339), (.492, .341), (.480, .349)]
    draw.polygon([(u*n, (1-z/.48)*n) for u,z in nose], fill=brown)
    draw.ellipse(box(.494, .352, .008, .0025), fill=(116, 77, 50))
    line([(.5, .340), (.5, .328)], brown, 8)
    for sign in (-1, 1):
        t = np.linspace(0, 1, 20)
        line([(.5+sign*.033*v, .328-.007*np.sin(v*np.pi)) for v in t], brown, 7)
    # Fine hand-stitched belly seam, kept subtle against the cream fabric.
    for theta in np.linspace(0, 2*np.pi, 110, endpoint=False):
        u = .5 + .136*np.cos(theta)
        z = .180 + .101*np.sin(theta)
        line([(u-.0012, z-.0007), (u+.0012, z+.0007)], (189, 151, 104), 2)
    # Retain a light textile grain on the painted patches.
    out = np.asarray(image).astype(float) + (gaussian_filter(noise,.5)*1.5)[...,None]
    Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save(ASSETS / "bear_fur.png")


def make_cotton_texture():
    n = 1024
    y, x = np.mgrid[:n, :n]
    rng = np.random.default_rng(24)
    grain = rng.normal(0, 1.2, (n, n))
    weave = (np.cos(x*np.pi) + np.cos(y*np.pi)) * 2
    rgb = np.array([229, 220, 196]) + (grain+weave)[...,None]
    im = Image.fromarray(np.clip(rgb,0,255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # A woven teal hem and sparse stitches distinguish the rectangular boundary.
    d.rectangle((3,3,n-4,n-4), outline=(114,153,151), width=9)
    for pos in range(20, n-20, 24):
        for yy in (17,n-18):
            d.line((pos,yy,pos+10,yy),fill=(157,161,143),width=2)
        for xx in (17,n-18):
            d.line((xx,pos,xx,pos+10),fill=(157,161,143),width=2)
    # Complementary sewn hook/loop zones on the original rectangle's short ends.
    # Colour marks the zone on both rendered faces; activation side is physical config.
    strip = round(n * .04 / .86)
    inset = round(n * .025 / .86)
    d.rectangle((inset,0,inset+strip,n-1),fill=(39,68,83))
    d.rectangle((n-inset-strip,0,n-inset-1,n-1),fill=(139,83,43))
    for yy in range(8,n-8,14):
        for xx in range(7,strip-5,12):
            xl=inset+xx
            d.arc((xl,yy,xl+6,yy+7),0,265,fill=(113,156,169),width=2)
            d.line((xl+6,yy+4,xl+6,yy+10),fill=(113,156,169),width=1)
            xr=n-inset-strip+xx
            d.ellipse((xr,yy,xr+6,yy+8),outline=(205,157,100),width=2)
    for xx in (inset+2,inset+strip-2,n-inset-strip+2,n-inset-2):
        for yy in range(5,n-5,14):
            d.line((xx,yy,xx,yy+7),fill=(232,222,196),width=2)
    im.save(ASSETS / "cotton_weave.png")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    spacing = .003
    low = np.array([-.17,-.16,-.012])
    high = np.array([.17,.14,.510])
    axes = [np.arange(a,b+spacing/2,spacing) for a,b in zip(low,high)]
    grid = np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1)
    vertices, faces, _, _ = marching_cubes(bear_field(grid), level=0, spacing=(spacing,)*3)
    vertices += low
    vertices[:, 2] = np.maximum(vertices[:,2],0)
    vertices *= .48 / vertices[:,2].max()
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    trimesh.repair.fix_normals(mesh)
    assert mesh.is_watertight and len(mesh.split()) == 1
    # A modest physical mesh retains the ears and face while leaving room for
    # interior tetrahedral nodes. Visual mesh stays smoother than the particles.
    pv,pf = fast_simplification.simplify(mesh.vertices, mesh.faces, target_count=1800, agg=5.)
    pv[:,2] = np.maximum(pv[:,2],0)
    physical = trimesh.Trimesh(vertices=pv,faces=pf,process=True)
    trimesh.repair.fix_normals(physical)
    assert physical.is_watertight
    tet = tetgen.TetGen(physical.vertices, physical.faces.astype(np.int32))
    points,tetra,_,_ = tet.tetrahedralize(quality=True,minratio=1.5,mindihedral=5,
                                       nobisect=True,maxvolume=8e-6,fixedvolume=True,
                                       steinerleft=1200,quiet=True)
    volumes = np.linalg.det(points[tetra[:,1:]]-points[tetra[:,:1]])/6
    negative = volumes<0
    tetra[negative] = tetra[negative][:,[0,2,1,3]]
    assert np.abs(volumes).min() > 1e-14 and len(points) <= 2500
    used,surface_faces = tetra_boundary(points,tetra)
    surface_points = points[used]
    shell = trimesh.Trimesh(vertices=surface_points,faces=surface_faces,process=False)
    assert shell.is_watertight and shell.is_winding_consistent and len(shell.split()) == 1
    assert np.isclose(shell.volume,np.abs(volumes).sum(),rtol=1e-8)
    # Keep z=0 planar vertices plus a 12 mm collar rigidly attached to the table.
    fixed_indices = np.flatnonzero(points[:,2] <= .012).astype(np.int32)
    vv,vf = fast_simplification.simplify(mesh.vertices,mesh.faces,target_count=9000,agg=5.)
    uv = cylindrical_uv(vv)
    write_obj(ASSETS/"bear.obj",vv,vf,uv,"bear_fur.png",seam=True)
    np.savez_compressed(ASSETS/"bear.npz",points=points,tetra=tetra,
                        surface_points=surface_points,surface_faces=surface_faces,
                        surface_point_indices=used,visual_points=vv,visual_faces=vf,uv=uv,
                        fixed_indices=fixed_indices)
    # Belly circumference is measured on the actual mesh at its widest torso
    # height; 12% overlap leaves enough length to encircle the soft abdomen.
    section = shell.section(plane_origin=(0,0,.17),plane_normal=(0,0,1))
    circumference = float(section.length)
    length = float(np.ceil(circumference*1.12/.01)*.01)
    width = .20
    nx,ny = int(np.ceil(length/.0125)),int(np.ceil(width/.0125))
    xs,ys = np.linspace(-length/2,length/2,nx+1),np.linspace(-width/2,width/2,ny+1)
    xx,yy = np.meshgrid(xs,ys,indexing="xy")
    cloth_points = np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
    cloth_faces=[]
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i; b=a+1; c=a+nx+1; d=c+1
            cloth_faces.extend(([a,b,d],[a,d,c]) if (i+j)%2==0 else ([a,b,c],[b,d,c]))
    cloth_faces=np.asarray(cloth_faces,dtype=np.int32)
    cloth_uv=np.column_stack([(cloth_points[:,0]+length/2)/length,
                              (cloth_points[:,1]+width/2)/width])
    write_obj(ASSETS/"cotton.obj",cloth_points,cloth_faces,cloth_uv,"cotton_weave.png")
    np.savez_compressed(ASSETS/"cotton.npz",points=cloth_points,faces=cloth_faces,uv=cloth_uv,
                        left_edge_indices=np.flatnonzero(np.isclose(cloth_points[:,0],-length/2)),
                        right_edge_indices=np.flatnonzero(np.isclose(cloth_points[:,0],length/2)))
    make_bear_texture()
    make_cotton_texture()
    metadata={
        "units":"metres", "bear_front":"negative Y", "bear_up":"positive Z",
        "bear": {"height_m":float(np.ptp(points[:,2])),"bounds_m":[points.min(0).tolist(),points.max(0).tolist()],
                 "physics_vertices":len(points),"tetrahedra":len(tetra),
                 "surface_vertices":len(surface_points),"surface_faces":len(surface_faces),
                 "visual_vertices":len(vv),"visual_faces":len(vf),"volume_m3":float(np.abs(volumes).sum()),
                 "watertight":bool(shell.is_watertight),"connected_components":len(shell.split()),
                 "fixed_base_vertices":len(fixed_indices),"fixed_base_z_max_m":.012,
                 "belly_circumference_m":circumference,"belly_measurement_z_m":.17,
                 "mesh":"bear.obj","physics":"bear.npz","texture":"bear_fur.png",
                 "features":["armless","flat fixed base","round ears","muzzle","two small feet"]},
        "cotton": {"length_m":length,"width_m":width,"area_m2":length*width,
                   "belly_overlap_fraction":length/circumference-1,
                   "vertices":len(cloth_points),"faces":len(cloth_faces),
                   "grid_spacing_m":[length/nx,width/ny],"mesh":"cotton.obj",
                   "physics":"cotton.npz","texture":"cotton_weave.png",
                   "areal_density_kg_m2":.180,"suggested_mass_kg":length*width*.180},
        "provenance":"Original deterministic procedural geometry and textures; generated locally.",
        "velcro":{"strip_width_m":.04,"strip_length_m":.20,"strip_inset_m":.025,
                  "locations":"original left/right short-end regions of same cotton mesh",
                  "visual":"navy hook / ochre loop zones; activation sides set by config",
                  "active_faces":"both original +Z faces, meeting as a folded seam",
                  "model":"surface bond zones, no additional thickness or mass"},
    }
    (ASSETS/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
