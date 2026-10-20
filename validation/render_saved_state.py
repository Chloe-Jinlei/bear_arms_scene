#!/usr/bin/env python3
"""Render an immutable saved state with native Genesis cameras and zero steps.

Example:
  python validation/render_saved_state.py /path/state-press.npz \
      --output /path/new-render-directory --backend cuda

By default use the state's sibling source-at-run, including its assets. State
metadata supplies the physical configuration. Only camera poses are added.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing outputs are never overwritten")
    parser.add_argument("--scene", type=Path,
                        help="Scene implementation/assets; defaults to sibling source-at-run")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--cameras", type=Path,
                        help="Optional JSON mapping camera names to position/target/fov")
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional shared compilation cache, separate from immutable inputs")
    args = parser.parse_args()
    if min(args.width, args.height) < 32:
        parser.error("Image dimensions must be at least 32")
    state_path = args.state.resolve()
    sibling = state_path.parent / "source-at-run"
    scene_dir = (args.scene or (sibling if (sibling / "assets").is_dir()
                               else Path(__file__).resolve().parents[1])).resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory")

    sys.dont_write_bytecode = True
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cache = args.cache_dir.resolve() if args.cache_dir else output / ".cache"
    for key, suffix in (("NUMBA_CACHE_DIR", "numba"), ("GS_CACHE_FILE_PATH", "genesis"),
                        ("XDG_CACHE_HOME", ""), ("MPLCONFIGDIR", "matplotlib")):
        os.environ[key] = str(cache / suffix)
    sys.path.insert(0, str(scene_dir))
    import numpy as np
    from PIL import Image

    initial_state_hash = sha256(state_path)
    with np.load(state_path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"]))
        state = {name: saved[name].copy() for name in
                 ("bear_positions", "cloth_positions", "left_qpos", "right_qpos")}
    config = copy.deepcopy(meta["config"])
    expected_assets = meta.get("asset_sha256", {})
    if not expected_assets:
        raise RuntimeError("Saved state has no asset hashes; cannot establish matching render assets")
    asset_hashes = {}
    for name, expected in expected_assets.items():
        path = scene_dir / "assets" / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Saved-state asset mismatch: {path}")
        asset_hashes[name] = expected

    cloth_positions = state["cloth_positions"]
    active = [int(b["source_node"]) for b in meta.get("velcro", {}).get("bonds", [])
              if b.get("active")]
    if active:
        center = np.median(cloth_positions[active], axis=0)
        center_method = "median of actual active bond source positions"
    else:
        center = np.array([config["bear"]["position_xy"][0],
                           config["bear"]["position_xy"][1] + .15,
                           config["table"]["height"] + .22])
        center_method = "configured rear belly fallback; no active bonds recorded"
    target = [float(center[0]), float(center[1]), float(center[2] - .015)]
    table = config["table"]["height"]
    camera_defs = {
        "rear_low": {"position": [float(center[0]), float(center[1] + .73), table + .12],
                     "target": target, "fov": 28.},
        "rear_left_low": {"position": [float(center[0] - .48), float(center[1] + .58), table + .21],
                          "target": target, "fov": 30.},
        "rear_right_low": {"position": [float(center[0] + .48), float(center[1] + .58), table + .21],
                           "target": target, "fov": 30.},
    }
    if args.cameras:
        camera_defs = json.loads(args.cameras.read_text())
    if not isinstance(camera_defs, dict) or not camera_defs:
        raise ValueError("Cameras must be a nonempty mapping")
    for name, camera in camera_defs.items():
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Camera names must be safe file-name components")
        if (len(camera["position"]) != 3 or len(camera["target"]) != 3
                or not np.isfinite([*camera["position"], *camera["target"], camera["fov"]]).all()
                or not 1 < camera["fov"] < 150):
            raise ValueError("Invalid camera geometry")
    physical_config = copy.deepcopy(config)
    config["cameras"].update(camera_defs)
    output.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("saved_state_render_scene", scene_dir / "scene.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = {
        "status": "initializing",
        "scope": "Static native Genesis render of saved positions and robot qpos; not a simulation replay or new action evidence",
        "state": str(state_path), "state_sha256": initial_state_hash,
        "stage": meta.get("phase"), "saved_step": meta.get("step"),
        "scene_source": str(scene_dir), "scene_py_sha256": sha256(scene_dir / "scene.py"),
        "renderer_script_sha256": sha256(__file__),
        "asset_sha256": asset_hashes, "physical_config_from_state_metadata": physical_config,
        "camera_target_method": center_method, "active_bonds_in_saved_metadata": len(active),
        "cameras": camera_defs, "resolution": [args.width, args.height],
        "robots_visible": True, "objects_hidden": [], "simulation_steps_requested": 0,
        "initialization_setters": ["robot.set_qpos", "bear.set_particles_pos", "cloth.set_particles_pos"],
        "no_dynamics_or_contact_solve_after_saved_state_initialization": True,
        "native_build_warmup": "Genesis 1.4.3 scene.build internally advances and resets the initial scene before loading the saved state; no saved-state motion is simulated",
    }
    manifest = output / "render_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    build_args = SimpleNamespace(backend=args.backend, width=args.width, height=args.height,
                                 headless=True, task="wrap")
    started = time.perf_counter()
    gs = None
    try:
        data, mesh_dir = module.read_assets(config, output)
        scene, robots, bear, cloth, cameras, _, _ = module.build_scene(
            config, data, mesh_dir, build_args, report)
        import genesis as gs
        start_step = int(scene.sim.cur_step_global)

        def forbidden_step(*unused_args, **unused_kwargs):
            raise RuntimeError("This renderer must never advance the simulation")

        scene.step = forbidden_step
        for side, robot in robots.items():
            robot.set_qpos(state[side + "_qpos"])
        bear.set_particles_pos(state["bear_positions"])
        cloth.set_particles_pos(state["cloth_positions"])
        actual = {"bear_positions": bear.get_particles_pos().cpu().numpy().copy(),
                  "cloth_positions": cloth.get_particles_pos().cpu().numpy().copy(),
                  **{s + "_qpos": r.get_qpos().cpu().numpy().copy() for s, r in robots.items()}}
        errors = {k: float(np.max(np.abs(actual[k] - state[k]))) for k in state}
        if any(e > 2e-7 for e in errors.values()):
            raise RuntimeError(f"Initialization does not reproduce saved state: {errors}")
        images = {}
        for name in camera_defs:
            # force_render refreshes visual fields despite the unchanged step
            # counter. It does not execute dynamics or any contact constraint.
            rgb, *_ = cameras[name].render(force_render=True)
            path = output / f"{name}.png"
            Image.fromarray(np.asarray(rgb)).save(path)
            images[name] = {"file": path.name, "sha256": sha256(path)}
        final = {"bear_positions": bear.get_particles_pos().cpu().numpy(),
                 "cloth_positions": cloth.get_particles_pos().cpu().numpy(),
                 **{s + "_qpos": r.get_qpos().cpu().numpy() for s, r in robots.items()}}
        render_changes = {k: float(np.max(np.abs(final[k] - actual[k]))) for k in actual}
        end_step = int(scene.sim.cur_step_global)
        if end_step != start_step or any(render_changes.values()) or sha256(state_path) != initial_state_hash:
            raise RuntimeError("State changed during a render-only operation")
        report.update(status="rendered_without_simulation", images=images,
                      initialization_max_abs_errors=errors,
                      render_max_abs_state_changes=render_changes,
                      native_step_counter_before=start_step, native_step_counter_after=end_step,
                      input_state_file_unchanged=True, elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        report.update(status="render_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        if gs is not None:
            gs.destroy()
    print(json.dumps({"status": report["status"], "output": str(output),
                      "state_sha256": initial_state_hash, "images": list(camera_defs)}, indent=2))


if __name__ == "__main__":
    main()
