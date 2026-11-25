"""ArUco markers stuck on curved surfaces: marker images + textured meshes that follow the surface.

A marker is printed on a flat sheet (black square of side `size`, plus a one-cell white quiet zone) and stuck on the
surface, so it bends with it:
    * cylinders (vertical columns, horizontal pipes): the sheet is unrolled onto the cylinder exactly (isometric -
      a developable surface), so the printed square keeps its size;
    * ellipsoids (doubly curved, not developable): the sheet is projected onto the surface along the marker normal
      (a decal); stretch is < 2 % for the sizes used here.
The mesh sits 1.5 mm off the surface (no z-fighting), is visual only (no collisions) and lives in geom group 1, so
the cameras see it. Image orientation: the marker's "up" is the world-up direction projected on the surface
(around the pipe for a pipe marker), its "right" is up x normal (as seen from outside).

    p, n, e_u, e_v = marker_frame(world, marker)       # ground-truth centre, outward normal, image right / up axes
    add_markers(spec, world)                           # called by World.add_to
"""
import cv2
import mujoco
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
CELLS = 6            # 4x4 bits + 1-cell black border
PX_PER_CELL = 32
OFFSET = 0.0015      # mesh offset along the normal [m]


def dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT)


def marker_image(marker_id):
    """Greyscale marker with a one-cell white quiet zone: (CELLS + 2) * PX_PER_CELL pixels square."""
    img = cv2.aruco.generateImageMarker(dictionary(), int(marker_id), CELLS * PX_PER_CELL)
    return cv2.copyMakeBorder(img, PX_PER_CELL, PX_PER_CELL, PX_PER_CELL, PX_PER_CELL, cv2.BORDER_CONSTANT, value=255)


def _is_cylinder(surf):
    return hasattr(surf, "axis_point") and not hasattr(surf, "radii")


def marker_frame(world, mk):
    """Ground truth: centre p (on the surface), outward normal n, image axes e_u (right) and e_v (up)."""
    surf = world.surface(mk.surface)
    if mk.s is not None:                      # pipe: axial position + angle around the axis
        p, n = surf.surface_point(mk.s, np.radians(mk.theta_deg))
    else:                                     # vertical cylinder / ellipsoid: azimuth + height
        a = np.radians(mk.azimuth_deg)
        dxy = np.array([np.cos(a), np.sin(a), 0.0])
        c = np.asarray(surf.center if hasattr(surf, "center") else (*surf.xy, 0.0), float)
        o = np.array([c[0], c[1], mk.z]) + 5.0 * dxy
        t = surf.ray(o, -dxy)
        if t is None:
            raise ValueError(f"marker {mk.id}: height {mk.z} m is off the surface {mk.surface}")
        p = o - t * dxy
        n = surf.normal_at(p)
    up = np.array([0.0, 0.0, 1.0]) - n[2] * n
    if np.linalg.norm(up) < 1e-3:             # marker facing straight up: image up along x
        up = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e_v = up / np.linalg.norm(up)
    e_u = np.cross(e_v, n)
    return p, n, e_u, e_v


def _surface_point(surf, p0, n0, e_u, e_v, u, v):
    """Where the sheet point (u, v) (metres from the marker centre, flat-sheet coordinates) lands on the surface;
    returns (point, normal)."""
    if _is_cylinder(surf):
        a = np.asarray(surf.axis, float)
        c = np.cross(a, n0)                                   # circumferential direction at the centre
        t = u * e_u + v * e_v
        ta, tc = t @ a, t @ c                                 # unroll: axial length + arc length
        ang = tc / surf.radius
        n = np.cos(ang) * n0 + np.sin(ang) * c
        ap = surf.axis_point() + ((p0 - surf.axis_point()) @ a) * a
        return ap + ta * a + surf.radius * n, n
    q = p0 + u * e_u + v * e_v + 0.3 * n0                     # decal: project along -n0
    t = surf.ray(q, -n0)
    p = q - t * n0
    return p, surf.normal_at(p)


def marker_mesh(world, mk, n_grid=16):
    """Vertices (N, 3), texture coords (N, 2), faces (F, 3) of the sheet (black square + quiet zone)."""
    surf = world.surface(mk.surface)
    p0, n0, e_u, e_v = marker_frame(world, mk)
    side = mk.size * (CELLS + 2) / CELLS
    g = np.linspace(-side / 2, side / 2, n_grid + 1)
    V, T = [], []
    for v in g:
        for u in g:
            p, n = _surface_point(surf, p0, n0, e_u, e_v, u, v)
            V.append(p + OFFSET * n)
            T.append([(u + side / 2) / side, 1.0 - (v + side / 2) / side])   # image row 0 = top of the marker
    F = []
    k = n_grid + 1
    for i in range(n_grid):
        for j in range(n_grid):
            a, b, c, d = i * k + j, i * k + j + 1, (i + 1) * k + j + 1, (i + 1) * k + j
            F += [[a, b, c], [a, c, d]]
    return np.array(V), np.array(T), np.array(F)


def add_markers(spec, world):
    """One textured, non-colliding mesh per ArucoMarker (group 1). Checks one marker per surface."""
    seen = {}
    for mk in world.markers:
        if mk.surface in seen:
            raise ValueError(f"surface {mk.surface!r} carries two markers ({seen[mk.surface]}, {mk.id})")
        seen[mk.surface] = mk.id
        name = f"aruco_{mk.id}"
        img = marker_image(mk.id)
        tex = spec.add_texture()
        tex.name, tex.type = name, mujoco.mjtTexture.mjTEXTURE_2D
        tex.width, tex.height, tex.nchannel = img.shape[1], img.shape[0], 3
        tex.data = np.repeat(img[:, :, None], 3, 2).tobytes()
        mat = spec.add_material()
        mat.name = name
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        mat.specular, mat.shininess = 0.05, 0.1                  # matte print
        V, T, F = marker_mesh(world, mk)
        me = spec.add_mesh()
        me.name = name
        me.uservert, me.usertexcoord, me.userface = V.ravel().tolist(), T.ravel().tolist(), F.ravel().tolist()
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        geom = spec.worldbody.add_geom()
        geom.name, geom.type, geom.meshname, geom.material = name, mujoco.mjtGeom.mjGEOM_MESH, name, name
        geom.contype = geom.conaffinity = 0
        geom.group = 1
