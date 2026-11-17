"""Generates the algorithm figures in docs/assets/fig/ (run: python docs/_figures/gen_figures.py).

The figures run small, simplified Python ports of meshoptimizer's clustering algorithms on a
jittered 2D triangle grid, so what they show is what the algorithms do, not a sketch:

- greedy growth (meshopt_buildMeshletsFlex / meshopt_buildMeshlets): topological priority,
  then distance to the meshlet centroid, seeds when a meshlet is full
- scan (meshopt_buildMeshletsScan): cut the index buffer in order
- spatial (meshopt_buildMeshletsSpatial): recursive SAH splits over sorted centroids
- partition (meshopt_partitionClusters): agglomerative merging of clusters by shared vertices
- the boundary locks clusterlod.h places between groups before simplifying them

In 2D every triangle faces the same way, so the normal-cone term of the greedy score drops out.
"""

import heapq
import math
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fig")

COLS, ROWS, CELL = 16, 11, 30.0
MAX_V, MAX_T, MIN_T = 16, 18, 8

# ---------------------------------------------------------------- mesh


def make_mesh(seed=7):
    rng = random.Random(seed)
    pts = []
    for j in range(ROWS + 1):
        for i in range(COLS + 1):
            jx = 0 if i in (0, COLS) else rng.uniform(-0.28, 0.28)
            jy = 0 if j in (0, ROWS) else rng.uniform(-0.28, 0.28)
            pts.append(((i + jx) * CELL, (j + jy) * CELL))
    tris = []
    vid = lambda i, j: j * (COLS + 1) + i
    for j in range(ROWS):
        for i in range(COLS):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            if (i + j) % 2 == 0:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]
    return pts, tris


def centroid(pts, t):
    return (sum(pts[v][0] for v in t) / 3, sum(pts[v][1] for v in t) / 3)


def area(pts, t):
    (ax, ay), (bx, by), (cx, cy) = (pts[v] for v in t)
    return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2


# ---------------------------------------------------------------- greedy (Flex with split_factor 0)


def build_greedy(pts, tris, snapshot_meshlet=None):
    n = len(tris)
    adj = {}
    for ti, t in enumerate(tris):
        for v in t:
            adj.setdefault(v, []).append(ti)
    live = {v: len(l) for v, l in adj.items()}
    cen = [centroid(pts, t) for t in tris]
    mesh_area = sum(area(pts, t) for t in tris)
    expected_radius = math.sqrt(mesh_area / n * 0.5 * MAX_T) * 0.5
    corner = (min(c[0] for c in cen), min(c[1] for c in cen))
    dist = lambda p, q: math.hypot(p[0] - q[0], p[1] - q[1])

    emitted = [False] * n
    meshlets = []
    cur_tris, cur_verts = [], []
    acc = [0.0, 0.0]
    seeds = []
    snapshot = None

    def pick_neighbor():
        center = (acc[0] / len(cur_tris), acc[1] / len(cur_tris))
        best, best_pri, best_score, info = None, 5, float("inf"), {}
        used = set(cur_verts)
        for v in cur_verts:
            for ti in adj[v]:
                a, b, c = tris[ti]
                extra = sum(x not in used for x in (a, b, c))
                if extra == 0:
                    pri = 0
                elif live[a] == 1 or live[b] == 1 or live[c] == 1:
                    pri = 1
                elif (live[a] == 2) + (live[b] == 2) + (live[c] == 2) >= 2:
                    pri = 1 + extra
                else:
                    pri = 2 + extra
                info[ti] = pri
                score = 1 + dist(cen[ti], center) / expected_radius
                if pri < best_pri or (pri == best_pri and score < best_score):
                    best, best_pri, best_score = ti, pri, score
        return best, center, info

    def pick_seed():
        nonlocal seeds
        seeds = [s for s in seeds if not emitted[s]]
        for v in cur_verts:
            cand = [ti for ti in adj[v]]
            if cand:
                best = min(cand, key=lambda ti: sum(live[x] for x in tris[ti]))
                seeds.append(best)
        if not seeds:
            return None
        return min(seeds, key=lambda ti: (sum(live[x] for x in tris[ti]), dist(cen[ti], corner)))

    first = min(range(n), key=lambda ti: dist(cen[ti], corner))
    while True:
        if not cur_tris and not meshlets:
            best = first
        else:
            best, center, info = pick_neighbor() if cur_tris else (None, None, {})
            if snapshot_meshlet is not None and len(meshlets) == snapshot_meshlet and len(cur_tris) == MAX_T // 2:
                snapshot = (list(cur_tris), list(cur_verts), dict(info), best, center, [list(m) for m in meshlets])
            if best is None:
                rest = [ti for ti in range(n) if not emitted[ti]]
                if not rest:
                    break
                c0 = center or corner
                best = min(rest, key=lambda ti: dist(cen[ti], c0))
        used = set(cur_verts)
        extra = sum(v not in used for v in tris[best])
        if cur_tris and (len(cur_verts) + extra > MAX_V or len(cur_tris) >= MAX_T):
            s = pick_seed()
            if s is not None:
                best = s
            meshlets.append(cur_tris)
            cur_tris, cur_verts, acc = [], [], [0.0, 0.0]
        for v in tris[best]:
            if v not in cur_verts:
                cur_verts.append(v)
        cur_tris.append(best)
        acc[0] += cen[best][0]
        acc[1] += cen[best][1]
        emitted[best] = True
        for v in tris[best]:
            adj[v].remove(best)
            live[v] -= 1
        if all(emitted):
            break
    if cur_tris:
        meshlets.append(cur_tris)
    return meshlets, snapshot


# ---------------------------------------------------------------- scan


def build_scan(tris):
    meshlets, cur, verts = [], [], set()
    for ti, t in enumerate(tris):
        extra = sum(v not in verts for v in t)
        if cur and (len(verts) + extra > MAX_V or len(cur) >= MAX_T):
            meshlets.append(cur)
            cur, verts = [], set()
        cur.append(ti)
        verts.update(t)
    if cur:
        meshlets.append(cur)
    return meshlets


# ---------------------------------------------------------------- spatial (SAH)


def divisible(count, lo, hi):
    return count >= lo if lo * 2 <= hi else count % lo <= (count // lo) * (hi - lo)


def build_spatial(pts, tris, record=None):
    boxes = []
    for t in tris:
        xs = [pts[v][0] for v in t]
        ys = [pts[v][1] for v in t]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    cen = [centroid(pts, t) for t in tris]
    leaves = []

    def half_perimeter(bs):
        x0 = min(b[0] for b in bs); y0 = min(b[1] for b in bs)
        x1 = max(b[2] for b in bs); y1 = max(b[3] for b in bs)
        return (x1 - x0) + (y1 - y0)

    def nverts(ids):
        return len({v for ti in ids for v in tris[ti]})

    def split(ids, depth):
        if len(ids) <= MAX_T and nverts(ids) <= MAX_V:
            leaves.append(ids)
            return
        best = None
        for axis in (0, 1):
            order = sorted(ids, key=lambda ti: cen[ti][axis])
            costs = []
            for i in range(MIN_T, len(order) - MIN_T + 1):
                if not divisible(i, MIN_T, MAX_T):
                    continue
                l, r = order[:i], order[i:]
                cost = half_perimeter([boxes[t] for t in l]) * len(l) + half_perimeter([boxes[t] for t in r]) * len(r)
                costs.append((i, cost))
                if best is None or cost < best[0]:
                    best = (cost, axis, i, order)
            if record is not None and depth == 0:
                record.append((axis, order, costs))
        if best is None:  # vertex bound and too small to split by triangle count: halve
            order = sorted(ids, key=lambda ti: cen[ti][0])
            best = (0, 0, len(order) // 2, order)
        _, axis, i, order = best
        if record is not None and depth == 0:
            record.append(("chosen", axis, i))
        split(order[:i], depth + 1)
        split(order[i:], depth + 1)

    split(list(range(len(tris))), 0)
    return leaves


# ---------------------------------------------------------------- partition (meshopt_partitionClusters)


def partition(tris, meshlets, target):
    cverts = [set(v for ti in m for v in tris[ti]) for m in meshlets]
    vref = {}
    for ci, vs in enumerate(cverts):
        for v in vs:
            vref.setdefault(v, []).append(ci)
    shared = {}
    for v, cs in vref.items():
        for a in cs:
            for b in cs:
                if a < b:
                    shared[(a, b)] = shared.get((a, b), 0) + 1
    group_of = list(range(len(meshlets)))
    members = {i: [i] for i in range(len(meshlets))}
    nv = {i: len(cverts[i]) for i in range(len(meshlets))}
    heap = [(nv[i], i) for i in members]
    heapq.heapify(heap)
    max_size = target + target // 3
    done = set()
    while heap:
        _, g = heapq.heappop(heap)
        if g not in members or g in done:
            continue
        if len(members[g]) >= target:
            done.add(g)
            continue
        acc = {}
        for c in members[g]:
            for (a, b), s in shared.items():
                o = b if a == c else a if b == c else None
                if o is None:
                    continue
                og = group_of[o]
                if og != g and og not in done:
                    acc[og] = acc.get(og, 0) + s
        best, best_score = None, 0
        for og, s in acc.items():
            if len(members[g]) + len(members[og]) > max_size:
                continue
            score = s * (1 / math.sqrt(nv[g]) + 1 / math.sqrt(nv[og]))
            if score > best_score:
                best, best_score = og, score
        if best is None:
            done.add(g)
            continue
        members[g] += members.pop(best)
        nv[g] = nv[g] + nv.pop(best) - acc[best]
        for c in members[g]:
            group_of[c] = g
        heapq.heappush(heap, (nv[g], g))
    return group_of, shared, cverts


# ---------------------------------------------------------------- svg helpers

PALETTE = ["#8ecae6", "#ffb703", "#90be6d", "#f28482", "#b8a1e3", "#f6bd60", "#84a59d", "#e5989b",
           "#a3c4f3", "#cdb4db", "#ffd6a5", "#9bf6ff", "#caffbf", "#fdffb6", "#ffc6ff", "#bdb2ff"]


def color_for(i):
    return PALETTE[(i * 7) % len(PALETTE)]


def color_meshlets(tris, meshlets):
    """Greedy graph coloring so neighboring meshlets get different colors."""
    owner = {}
    for mi, m in enumerate(meshlets):
        for ti in m:
            owner[ti] = mi
    vown = {}
    for mi, m in enumerate(meshlets):
        for ti in m:
            for v in tris[ti]:
                vown.setdefault(v, set()).add(mi)
    nb = [set() for _ in meshlets]
    for s in vown.values():
        for a in s:
            nb[a] |= s - {a}
    colors = []
    for mi in range(len(meshlets)):
        taken = {colors[o] for o in nb[mi] if o < mi}
        k = (mi * 5) % len(PALETTE)
        while PALETTE[k] in taken:
            k = (k + 1) % len(PALETTE)
        colors.append(PALETTE[k])
    return colors, owner


def poly(pts, t, **attrs):
    p = " ".join(f"{pts[v][0]:.1f},{pts[v][1]:.1f}" for v in t)
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polygon points="{p}" {a}/>'


def boundary_edges(tris, groups_of_tri):
    """Edges whose two triangles belong to different groups (or that lie on the mesh border)."""
    e = {}
    for ti, t in enumerate(tris):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            e.setdefault((min(a, b), max(a, b)), []).append(ti)
    out = []
    for (a, b), ts in e.items():
        if len(ts) == 1 or groups_of_tri[ts[0]] != groups_of_tri[ts[1]]:
            out.append((a, b))
    return out


def mesh_panel(pts, tris, fills, outline_groups=None, dx=0, dy=0, wire="#00000033", stroke_w=2.2, extra=""):
    s = [f'<g transform="translate({dx},{dy})">']
    for ti, t in enumerate(tris):
        s.append(poly(pts, t, fill=fills[ti], stroke=wire, stroke_width=0.6, stroke_linejoin="round"))
    if outline_groups is not None:
        for a, b in boundary_edges(tris, outline_groups):
            (x0, y0), (x1, y1) = pts[a], pts[b]
            s.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#1d1d1f" stroke-width="{stroke_w}" stroke-linecap="round"/>')
    s.append(extra)
    s.append("</g>")
    return "\n".join(s)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="13">\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n')


def label(x, y, text, size=14, weight="600", anchor="start", fill="#1d1d1f"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{text}</text>'


def write(name, content):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print("wrote", name)


# ---------------------------------------------------------------- figures


def fig_builders(pts, tris):
    W, H = COLS * CELL, ROWS * CELL
    pad, top = 20, 34
    panels = [("Scan: index order", build_scan(tris)),
              ("Greedy: grow by adjacency", build_greedy(pts, tris)[0]),
              ("Spatial: SAH splits", build_spatial(pts, tris))]
    body = []
    for k, (title, ms) in enumerate(panels):
        colors, owner = color_meshlets(tris, ms)
        fills = [colors[owner[ti]] for ti in range(len(tris))]
        dx = pad + k * (W + pad)
        body.append(label(dx, 22, title))
        body.append(mesh_panel(pts, tris, fills, [owner[ti] for ti in range(len(tris))], dx, top))
        avg_v = sum(len({v for ti in m for v in tris[ti]}) for m in ms) / len(ms)
        body.append(label(dx, top + H + 22, f"{len(ms)} meshlets, avg {len(tris) / len(ms):.1f} tris / {avg_v:.1f} verts",
                          size=12, weight="400", fill="#555"))
    write("builders.svg", svg(3 * W + 4 * pad, top + H + 34, "\n".join(body)))


def fig_greedy_step(pts, tris):
    target = 7
    ms, snap = build_greedy(pts, tris, snapshot_meshlet=target)
    cur_tris, cur_verts, info, best, center, done = snap
    W, H = COLS * CELL, ROWS * CELL
    pri_colors = {0: "#2a9d8f", 1: "#8ab17d", 2: "#e9c46a", 3: "#f4a261", 4: "#e76f51"}
    owner = {}
    for mi, m in enumerate(done):
        for ti in m:
            owner[ti] = mi
    fills = []
    for ti in range(len(tris)):
        if ti in cur_tris:
            fills.append("#4361ee")
        elif ti in owner:
            fills.append("#d9d9de")
        elif ti in info:
            fills.append(pri_colors[min(info[ti], 4)])
        else:
            fills.append("#ffffff")
    groups = [owner.get(ti, -2 if ti in cur_tris else -1) for ti in range(len(tris))]
    extra = []
    for ti, pri in info.items():
        cx, cy = centroid(pts, tris[ti])
        extra.append(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#1d1d1f">{pri}</text>')
    bx, by = centroid(pts, tris[best])
    extra.append(poly(pts, tris[best], fill="none", stroke="#d00000", stroke_width=2.5, stroke_linejoin="round"))
    for v in cur_verts:
        x, y = pts[v]
        extra.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#ffffff" stroke="#1d1d1f" stroke-width="1"/>')
    cx, cy = center
    extra.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#ffffff" stroke="#d00000" stroke-width="2"/>')
    extra.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#d00000" stroke-width="1.5" stroke-dasharray="4 3"/>')
    pad, top = 20, 34
    body = [label(pad, 22, f"Growing meshlet {target + 1}: {len(cur_tris)} triangles, {len(cur_verts)}/{MAX_V} vertices"),
            mesh_panel(pts, tris, fills, groups, pad, top, extra="\n".join(extra))]
    lx = pad + W + 30
    legend = [("#4361ee", "current meshlet"), ("#d9d9de", "earlier meshlets")]
    legend += [(pri_colors[p], f"candidate, priority {p}") for p in range(5)]
    for i, (c, t) in enumerate(legend):
        y = top + 10 + i * 24
        body.append(f'<rect x="{lx}" y="{y}" width="16" height="16" rx="3" fill="{c}" stroke="#00000044"/>')
        body.append(label(lx + 24, y + 13, t, size=12, weight="400"))
    y = top + 10 + len(legend) * 24 + 12
    notes = ["0  adds no new vertex",
             "1  uses a vertex with 1 live triangle",
             "   (would be left dangling)",
             "2-4  1 + or 2 + new vertices",
             "",
             "red outline: chosen next",
             "red circle: meshlet centroid;",
             "ties broken by distance to it"]
    for i, t in enumerate(notes):
        body.append(label(lx, y + i * 17, t, size=11.5, weight="400", fill="#333"))
    write("greedy-step.svg", svg(W + 2 * pad + 250, top + H + 20, "\n".join(body)))


def fig_sah(pts, tris):
    rec = []
    build_spatial(pts, tris, record=rec)
    W, H = COLS * CELL, ROWS * CELL
    chosen = [r for r in rec if r[0] == "chosen"][0]
    _, axis, split_at = chosen
    axes = [r for r in rec if r[0] != "chosen"]
    order = [r for r in axes if r[0] == axis][0][1]
    left = set(order[:split_at])
    fills = ["#a3c4f3" if ti in left else "#ffd6a5" for ti in range(len(tris))]
    groups = [0 if ti in left else 1 for ti in range(len(tris))]
    pad, top = 20, 34
    body = [label(pad, 22, "First split: the cheapest cut over both axes"),
            mesh_panel(pts, tris, fills, groups, pad, top, stroke_w=3)]
    # cost curves
    gx, gy, gw, gh = pad + W + 50, top + 10, 300, H - 50
    all_costs = [c for r in axes for _, c in r[2]]
    cmin, cmax = min(all_costs), max(all_costs)
    n = len(tris)
    body.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="none" stroke="#ccc"/>')
    for r, col, name in ((axes[0], "#4361ee", "split along x"), (axes[1], "#e76f51", "split along y")):
        pts_s = " ".join(f"{gx + i / n * gw:.1f},{gy + gh - (c - cmin) / (cmax - cmin) * (gh - 10) - 5:.1f}" for i, c in r[2])
        body.append(f'<polyline points="{pts_s}" fill="none" stroke="{col}" stroke-width="2"/>')
    for i, (col, name) in enumerate((("#4361ee", "split along x"), ("#e76f51", "split along y"))):
        body.append(f'<line x1="{gx + 10}" y1="{gy + 16 + i * 18}" x2="{gx + 30}" y2="{gy + 16 + i * 18}" stroke="{col}" stroke-width="2"/>')
        body.append(label(gx + 36, gy + 20 + i * 18, name, size=12, weight="400"))
    ci = [c for i, c in axes[axis][2] if i == split_at][0]
    mx = gx + split_at / n * gw
    my = gy + gh - (ci - cmin) / (cmax - cmin) * (gh - 10) - 5
    body.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="5" fill="none" stroke="#1d1d1f" stroke-width="2"/>')
    body.append(label(gx + gw / 2, gy + gh + 20, "triangles on the left side of the cut", size=12, weight="400", anchor="middle"))
    body.append(f'<text transform="translate({gx - 12},{gy + gh / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#1d1d1f">SAH cost</text>')
    write("sah.svg", svg(gx + gw + 20, top + H + 20, "\n".join(body)))


def fig_partition(pts, tris):
    ms, _ = build_greedy(pts, tris)
    group_of, shared, cverts = partition(tris, ms, target=4)
    W, H = COLS * CELL, ROWS * CELL
    colors, owner = color_meshlets(tris, ms)
    pad, top = 20, 34
    cc = []
    for m in ms:
        xs = [centroid(pts, tris[ti]) for ti in m]
        cc.append((sum(x for x, _ in xs) / len(xs), sum(y for _, y in xs) / len(xs)))
    # panel 1: cluster adjacency graph
    fills1 = ["#f1f1f4"] * len(tris)
    for ti in range(len(tris)):
        fills1[ti] = colors[owner[ti]] + "88"
    g = []
    for (a, b), s in shared.items():
        g.append(f'<line x1="{cc[a][0]:.1f}" y1="{cc[a][1]:.1f}" x2="{cc[b][0]:.1f}" y2="{cc[b][1]:.1f}" stroke="#1d1d1f" stroke-opacity="0.7" stroke-width="{0.5 + s * 0.45:.2f}"/>')
    for x, y in cc:
        g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#1d1d1f"/>')
    body = [label(pad, 22, "Clusters and their adjacency (line width = shared vertices)"),
            mesh_panel(pts, tris, fills1, [owner[ti] for ti in range(len(tris))], pad, top, wire="#00000018", extra="\n".join(g))]
    # panel 2: groups with locked boundary vertices
    gids = sorted(set(group_of))
    group_palette = ["#8ecae6", "#ffb703", "#90be6d", "#f28482", "#b8a1e3", "#84a59d", "#f6bd60", "#a3c4f3"]
    gcol = {gid: group_palette[k % len(group_palette)] for k, gid in enumerate(gids)}
    tri_group = [group_of[owner[ti]] for ti in range(len(tris))]
    fills2 = [gcol[tri_group[ti]] for ti in range(len(tris))]
    cluster_lines = []
    for a, b in boundary_edges(tris, [owner[ti] for ti in range(len(tris))]):
        (x0, y0), (x1, y1) = pts[a], pts[b]
        cluster_lines.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#1d1d1f" stroke-opacity="0.45" stroke-width="1" stroke-dasharray="3 2"/>')
    vgroups = {}
    for ti, t in enumerate(tris):
        for v in t:
            vgroups.setdefault(v, set()).add(tri_group[ti])
    locked = [v for v, s in vgroups.items() if len(s) > 1]
    dots = "\n".join(cluster_lines) + "\n" + "\n".join(f'<circle cx="{pts[v][0]:.1f}" cy="{pts[v][1]:.1f}" r="3" fill="#d00000"/>' for v in locked)
    dx2 = pad + W + 40
    body.append(label(dx2, 22, f"{len(gids)} groups of ~4 clusters; red vertices are locked"))
    body.append(mesh_panel(pts, tris, fills2, tri_group, dx2, top, extra=dots, stroke_w=3))
    # thin cluster outlines inside groups
    write("partition.svg", svg(dx2 + W + pad, top + H + 20, "\n".join(body)))


def box(x, y, w, h, text, fill="#f4f6fb", stroke="#8a94a6", sub=None, rx=8):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>']
    ty = y + h / 2 + (5 if sub is None else -3)
    s.append(label(x + w / 2, ty, text, size=13, weight="600", anchor="middle"))
    if sub:
        s.append(label(x + w / 2, ty + 17, sub, size=11, weight="400", anchor="middle", fill="#555"))
    return "\n".join(s)


def arrow(x0, y0, x1, y1, color="#5c6370", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="1.6" marker-end="url(#arrow)"{d}/>'


ARROW_DEF = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#5c6370"/></marker></defs>')


def fig_pipeline():
    body = [ARROW_DEF]
    bw, bh, gap = 150, 54, 28
    # row 1: load + preprocess + level 0
    row1 = [("Load mesh", "PLY / OBJ"), ("Weld + normals", "generateVertexRemap"), ("Vertex cache order", "optimizeVertexCache"),
            ("Clusterize level 0", "buildMeshletsFlex")]
    x0, y1 = 20, 50
    body.append(label(20, 28, "Build (CPU, background thread)", size=14))
    for i, (t, s) in enumerate(row1):
        x = x0 + i * (bw + gap)
        body.append(box(x, y1, bw, bh, t, sub=s))
        if i:
            body.append(arrow(x - gap, y1 + bh / 2, x - 2, y1 + bh / 2))
    # LOD loop
    loop = [("Partition", "partitionClusters"), ("Lock group borders", "shared vertices"), ("Simplify group", "to 50% of triangles"),
            ("Re-clusterize", "buildMeshletsFlex")]
    y2 = y1 + bh + 56
    lx = [x0 + (3 - i) * (bw + gap) for i in range(4)]
    for i, (t, s) in enumerate(loop):
        body.append(box(lx[i], y2, bw, bh, t, sub=s, fill="#eef6ee", stroke="#7fa37f"))
        if i:
            body.append(arrow(lx[i - 1] - 2, y2 + bh / 2, lx[i] + bw + 2, y2 + bh / 2))
    body.append(arrow(lx[0] + bw / 2, y1 + bh + 2, lx[0] + bw / 2, y2 - 2))
    # loop back arrow from re-clusterize to partition
    yb = y2 + bh + 22
    body.append(f'<path d="M{lx[3] + bw / 2},{y2 + bh + 2} V{yb} H{lx[0] + bw / 2} V{y2 + bh + 4}" fill="none" stroke="#7fa37f" stroke-width="1.6" marker-end="url(#arrow)"/>')
    body.append(label((lx[0] + lx[3] + bw) / 2, yb + 18, "repeat per DAG level until one cluster is left or simplification gets stuck",
                      size=11.5, weight="400", anchor="middle", fill="#3f6b3f"))
    body.append(label(lx[3] - 6, y2 - 10, "clodBuild", size=12, weight="700", fill="#3f6b3f"))
    # finalize
    y3 = yb + 46
    fin = [("Per-meshlet data", "bounds, cone, edge masks"), ("Group BVH", "clodBuildHierarchy")]
    for i, (t, s) in enumerate(fin):
        x = x0 + i * (bw + gap)
        body.append(box(x, y3, bw, bh, t, sub=s))
        if i:
            body.append(arrow(x - gap, y3 + bh / 2, x - 2, y3 + bh / 2))
    body.append(arrow(lx[3] + 25, y2 + bh + 2, lx[3] + 25, y3 - 2))
    # GPU row
    y4 = y3 + bh + 70
    body.append(label(20, y4 - 16, "Render (GPU, every frame)", size=14))
    gpu = [("BVH traversal", "compute, 1 pass / depth"), ("Task shader", "LOD test + culling"), ("Mesh shader", "1 workgroup / meshlet"),
           ("Fragment shader", "color, outlines")]
    for i, (t, s) in enumerate(gpu):
        x = x0 + i * (bw + gap)
        body.append(box(x, y4, bw, bh, t, sub=s, fill="#fbf1e6", stroke="#c99a62"))
        if i:
            body.append(arrow(x - gap, y4 + bh / 2, x - 2, y4 + bh / 2))
    body.append(arrow(x0 + bw + gap + bw / 2, y3 + bh + 2, x0 + bw + gap + bw / 2, y4 - 2, dash="5 4"))
    body.append(label(x0 + bw + gap + bw / 2 + 8, y3 + bh + 38, "upload to GPU buffers", size=11.5, weight="400", fill="#555"))
    write("pipeline.svg", svg(x0 + 4 * bw + 3 * gap + 20, y4 + bh + 20, "\n".join(body)))


def fig_dag():
    W = 760
    left, right = 30, W - 30
    level_y = {0: 330, 1: 235, 2: 140, 3: 45}
    counts = {0: 16, 1: 8, 2: 4, 3: 2}
    bw, bh = 30, 30
    lvl_color = {0: "#3a6fd8", 1: "#2fa3a8", 2: "#7cbf6a", 3: "#e8a33d"}
    # groups: (level, first cluster, count, error, label)
    groups = [(0, 0, 4, "3.1"), (0, 4, 4, "0.6"), (0, 8, 4, "0.5"), (0, 12, 4, "0.7"),
              (1, 0, 4, "4.0"), (1, 4, 4, "0.9"), (2, 0, 4, "6.5"), (3, 0, 2, "∞")]
    selected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)}

    def cx(level, i):
        return left + (i + 0.5) * (right - left) / counts[level]

    body = [ARROW_DEF]
    for g, (level, first, n, err) in enumerate(groups):
        x0 = cx(level, first) - bw / 2 - 8
        x1 = cx(level, first + n - 1) + bw / 2 + 8
        y = level_y[level]
        over = err == "∞" or float(err) > 1
        body.append(f'<rect x="{x0:.1f}" y="{y - 8}" width="{x1 - x0:.1f}" height="{bh + 16}" rx="10" fill="none" '
                    f'stroke="{"#c0392b" if over else "#27ae60"}" stroke-width="1.6" stroke-dasharray="5 3"/>')
        body.append(label((x0 + x1) / 2, y - 14, f"group error {err} px", size=11, weight="600", anchor="middle",
                          fill="#c0392b" if over else "#27ae60"))
        if level < 3:
            # produced clusters at the next level: 2 per group of 4
            nxt = level + 1
            k0 = first // 2
            for k in range(k0, k0 + n // 2):
                body.append(f'<line x1="{(x0 + x1) / 2:.1f}" y1="{y - 22}" x2="{cx(nxt, k):.1f}" y2="{level_y[nxt] + bh + 2}" '
                            f'stroke="#9aa3b2" stroke-width="1.2" marker-end="url(#arrow)"/>')
    for level, n in counts.items():
        for i in range(n):
            on = (level, i) in selected
            fill = lvl_color[level] if on else "#e6e8ee"
            body.append(f'<rect x="{cx(level, i) - bw / 2:.1f}" y="{level_y[level]}" width="{bw}" height="{bh}" rx="5" fill="{fill}" '
                        f'stroke="{"#1d1d1f" if on else "#b8bdc8"}" stroke-width="{1.6 if on else 1}"/>')
        body.append(label(8, level_y[level] + 20, f"L{level}", size=12, weight="700", fill=lvl_color[level]))
    body.append(label(W - 20, 30, "threshold: 1 px", size=12, weight="700", anchor="end"))
    body.append(label(W - 20, 48, "filled = drawn this frame", size=11.5, weight="400", anchor="end", fill="#555"))
    write("dag.svg", svg(W, 380, "\n".join(body)))


def fig_layout():
    body = [ARROW_DEF]
    body.append(label(20, 24, "meshlets[i]", size=13))
    fields = [("vertex_offset", "120"), ("triangle_offset", "372"), ("vertex_count", "6"), ("triangle_count", "4")]
    for k, (n, v) in enumerate(fields):
        body.append(box(20, 36 + k * 30, 170, 26, f"{n} = {v}", rx=4))
    # vertices array
    vx, vy = 260, 40
    body.append(label(vx, 24, "vertices[]: meshlet-local → global vertex index (u32)", size=13))
    glob = [812, 813, 977, 978, 1140, 1141]
    for k in range(8):
        on = 1 <= k <= 6
        x = vx + k * 62
        body.append(f'<rect x="{x}" y="{vy}" width="58" height="30" rx="4" fill="{"#dce7fb" if on else "#f1f1f4"}" stroke="#8a94a6"/>')
        if on:
            body.append(label(x + 29, vy + 20, str(glob[k - 1]), size=12, weight="600", anchor="middle"))
            body.append(label(x + 29, vy + 46, str(k - 1), size=11, weight="400", anchor="middle", fill="#555"))
        else:
            body.append(label(x + 29, vy + 20, "…", size=12, weight="400", anchor="middle", fill="#999"))
    body.append(label(vx + 62 - 4, vy + 46, "local:", size=11, weight="400", anchor="end", fill="#555"))
    # triangles array
    ty = 130
    body.append(label(vx, ty - 12, "triangles[]: 3 local indices per triangle (u8)", size=13))
    tri = [(0, 1, 2), (1, 3, 2), (2, 3, 4), (3, 5, 4)]
    for k, t in enumerate(tri):
        for j, v in enumerate(t):
            x = vx + 62 + (k * 3 + j) * 30
            body.append(f'<rect x="{x}" y="{ty}" width="28" height="28" rx="3" fill="{["#fde2c8", "#d8f0d2"][k % 2]}" stroke="#8a94a6"/>')
            body.append(label(x + 14, ty + 19, str(v), size=12, weight="600", anchor="middle"))
    body.append(arrow(190, 49, vx + 62, vy + 15))
    body.append(arrow(190, 79, vx + 62, ty + 14))
    write("layout.svg", svg(760, 180, "\n".join(body)))


def main():
    pts, tris = make_mesh()
    fig_builders(pts, tris)
    fig_greedy_step(pts, tris)
    fig_sah(pts, tris)
    fig_partition(pts, tris)
    fig_pipeline()
    fig_dag()
    fig_layout()


if __name__ == "__main__":
    main()
