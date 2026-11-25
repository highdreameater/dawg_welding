"""Fast checks of the ArUco environment and its navigation components (no walking; ~1 min).

    python sim/aruco_selftest.py            # prints PASS / FAIL per check, exit code 1 on any failure

The full closed-loop checks (walking, stance confirmation, re-observation, termination) come from the mission:
`python sim/aruco_mission.py` writes them to results/aruco/summary_<robot>.json ("checks").
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markers import OFFSET, marker_frame, marker_mesh  # noqa: E402
from perception.aruco import ArucoDetector, RgbCamera, camera_spec  # noqa: E402
from planning.explorer import Explorer, _raster  # noqa: E402
from planning.grid_planner import OccupancyGrid  # noqa: E402
from planning.targets import MarkerRegistry, plan_to_marker  # noqa: E402
from wbc.robots import ROBOTS, build_model  # noqa: E402
from wbc.stance_wbc import Indexer, initial_state  # noqa: E402
from world import Cylinder, Ellipsoid, Pipe, aruco_workshop  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""), flush=True)


def surface_distance(surf, v):
    """Distance of a point from the surface (exact for cylinders, along the centre ray for ellipsoids)."""
    if isinstance(surf, Ellipsoid):
        c = np.asarray(surf.center, float)
        d = (v - c) / np.linalg.norm(v - c)
        return np.linalg.norm(v - c) - surf.ray(c, d)
    a, ap = surf.axis, surf.axis_point()
    w = (v - ap) - ((v - ap) @ a) * a
    return np.linalg.norm(w) - surf.radius


def main():
    world = aruco_workshop()
    cfg = ROBOTS["go2_z1"]()
    m = build_model(cfg, world=world, contact_tip=True, cameras=[camera_spec(cfg.name)])
    d = mujoco.MjData(m)
    ix = Indexer(m, cfg)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]

    # 1. four markers spawned (textured meshes, unique ids)
    geoms = [i for i, n in enumerate(names) if n.startswith("aruco_")]
    ids = [mk.id for mk in world.markers]
    check("1 four ArUco markers spawned", len(geoms) == 4 and len(set(ids)) == 4
          and all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_matid[g] >= 0 for g in geoms),
          f"ids {ids}, geoms {[names[g] for g in geoms]}")

    # 2. each on exactly one curved surface; the printed sheet lies on that surface
    surfs = [mk.surface for mk in world.markers]
    kinds = [type(world.surface(s)).__name__ for s in surfs]
    dev = []
    for mk in world.markers:
        V, _, _ = marker_mesh(world, mk)
        dev.append(max(abs(surface_distance(world.surface(mk.surface), v) - OFFSET) for v in V))
    check("2 one marker per cylindrical / curved surface", len(set(surfs)) == 4
          and all(isinstance(world.surface(s), (Cylinder, Pipe, Ellipsoid)) for s in surfs) and max(dev) < 2e-3,
          f"surfaces {dict(zip(surfs, kinds))}, sheet off the surface by <= {1000 * max(dev):.2f} mm (+1.5 mm offset)")

    # 3. detection + pose estimation from the robot's head camera (approach pose, and 1.6 m / 25 deg off-axis)
    cam, det = RgbCamera(m), None
    det = ArucoDetector(cam.K, 0.12)
    rows, ok3 = [], True
    for mk in world.markers:
        p, n, _, _ = marker_frame(world, mk)
        nh = n[:2] / np.linalg.norm(n[:2])
        for dist, off in ((0.75, 0.0), (1.6, 25.0)):
            a = np.radians(off)
            u = np.array([np.cos(a) * nh[0] - np.sin(a) * nh[1], np.sin(a) * nh[0] + np.cos(a) * nh[1]])
            initial_state(m, d, ix, xy=p[:2] + dist * u, yaw=np.arctan2(-u[1], -u[0]))
            obs = [o for o in det.detect(*cam.capture(m, d)) if o.id == mk.id]
            if not obs:
                ok3 = False
                rows.append(f"{mk.id}@{dist}m: missed")
                continue
            e_p, e_n = 1000 * np.linalg.norm(obs[0].p - p), np.degrees(np.arccos(np.clip(obs[0].n @ n, -1, 1)))
            ok3 &= e_p < 40 and e_n < 8
            rows.append(f"{mk.id}@{dist}m: {e_p:.0f} mm / {e_n:.1f} deg")
    check("3 marker detection + pose estimate (< 40 mm, < 8 deg; the strongly curved pipe marker is the worst)", ok3, "; ".join(rows))

    # 4/5. planning on the existing map: obstacles and stairs (stairs not traversable, real detour)
    grid = OccupancyGrid(world)
    stairs = np.zeros(grid.shape, bool)
    for ob in world.map_objects():
        if ob["kind"] == "stairs":
            stairs |= _raster(grid, ob["footprint"])
    order = [world.markers[i] for i in (1, 0, 2, 3)]
    xy, lens, ok45 = np.array(world.start[:2]), [], True
    for mk in order:
        p, n, _, _ = marker_frame(world, mk)
        try:
            path, info, pose, _, _ = plan_to_marker(grid, xy, p, n)
        except RuntimeError as e:
            ok45 = False
            lens.append(f"{mk.id}: {e}")
            continue
        on_stairs = any(stairs[grid.to_cell(q)] for q in path)
        ok45 &= not on_stairs and info["min_clearance"] >= 0.30
        lens.append(f"->{mk.id} {info['length']:.1f} m (min clearance {info['min_clearance']:.2f} m)")
        xy = pose[:2]
    check("4 navigation on the existing occupancy map (all targets reachable, clearance >= 0.30 m)", ok45,
          ", ".join(lens))
    w0 = aruco_workshop()
    w0.stairs = []
    p5, n5, _, _ = marker_frame(world, world.markers[0])      # column, planned from the vessel's approach pose
    p1, n1, _, _ = marker_frame(world, world.markers[1])
    start = plan_to_marker(grid, world.start[:2], p1, n1)[2][:2]
    L_st = plan_to_marker(grid, start, p5, n5)[1]["length"]
    L_no = plan_to_marker(OccupancyGrid(w0), start, p5, n5)[1]["length"]
    check("5 stairs are accounted for (vessel -> column detours around them)", L_st > L_no + 0.2,
          f"{L_st:.2f} m with stairs vs {L_no:.2f} m without; the RL height scan sees them as terrain (group 0)")

    # 8. exploration: at the sphere tank with every other surface already checked, explore until the hidden pipe
    #    marker is detected (teleporting to each chosen viewpoint and turning there in 12 camera frames)
    ex = Explorer(grid, world)
    pipe_mk = world.markers[3]
    tank = marker_frame(world, world.markers[2])
    for mk in world.markers[:3]:                             # pretend these surfaces were scanned from close up
        q, nn, _, _ = marker_frame(world, mk)
        ex.mark_seen(q[:2] + 0.75 * nn[:2] / np.linalg.norm(nn[:2]), 0.0, 2 * np.pi)
    xy = tank[0][:2] + 0.75 * np.array([0, -1.0])
    ex.mark_seen(xy, 0.0, 2 * np.pi)
    found, steps = False, []
    for _ in range(8):
        vp = ex.next_viewpoint(xy)
        if vp is None:
            break
        xy = vp["xy"]
        steps.append(np.round(xy, 2).tolist())
        for yaw in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            initial_state(m, d, ix, xy=xy, yaw=yaw)
            ex.mark_seen(cam.pose(d)[0][:2], yaw, cam.hfov)
            if any(o.id == pipe_mk.id for o in det.detect(*cam.capture(m, d))):
                found = True
        if found:
            break
    check("8 exploration finds a marker that is not visible", found, f"viewpoints {steps}")

    # 9. registry: visited markers are never selected again
    reg = MarkerRegistry(4, min_views=3)

    class O:  # minimal observation
        def __init__(self, i, p):
            self.id, self.p, self.n, self.dist, self.view_deg, self.ambiguity, self.t = i, np.array(p, float), \
                np.array([1.0, 0, 0]), 1.0, 0.0, 0.0, 0.0
    reg.update([O(1, (1, 0, 0)), O(2, (3, 0, 0)), O(17, (9, 9, 0))])       # 17: a one-frame false positive
    none_yet = reg.next_target((0, 0)) is None                             # < 3 views: still candidates
    for _ in range(2):
        reg.update([O(1, (1, 0, 0)), O(2, (3, 0, 0))])
    first = reg.next_target((0, 0)).id
    reg.mark_visited(first, 0.0)
    second = reg.next_target((0, 0)).id
    try:
        reg.mark_visited(first, 1.0)
        dup = False
    except RuntimeError:
        dup = True
    check("9 registry: no revisits, single-frame ids never become targets",
          none_yet and first == 1 and second == 2 and dup and 17 in reg.candidates() and 17 not in reg.order_detected,
          "confirmed after 3 views, nearest first, visited never re-selected, phantom id 17 stays a candidate")

    print(f"\n{sum(ok for _, ok in RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(ok for _, ok in RESULTS) else 1)


if __name__ == "__main__":
    main()
