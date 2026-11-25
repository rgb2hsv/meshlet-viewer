"""Quality statistics of .mlod meshlet hierarchies, for the builder comparison in docs/index.md.

    python docs/_figures/mlod_stats.py meshoptimizer.mlod meshlet-optimizer.mlod

Plain Python (no numpy). For every DAG level k it measures the surface the viewer draws with
--lod-level k (the level's clusters plus terminal clusters below it):

- clusters, triangles, average triangles and vertices per cluster, vertex references per triangle
  (the mesh shader transforms each referenced vertex once per cluster, so lower is cheaper);
- triangle shape: the radius ratio q = 4 sqrt(3) area / (sum of squared edge lengths), 1 for an
  equilateral triangle, 0 for a degenerate one; median, and the share of slivers (q < 0.3);
- cluster shape: the isoperimetric quotient 4 pi area / perimeter^2 of each cluster (1 for a disk);
  perimeter edges come from the file's edge masks;
- normal cones: the share of clusters whose cone allows backface culling;
- geometric deviation from the input (level 0), two-sided, in units of level 0's average edge
  length: level vertices to the level-0 surface and level-0 vertices to the level surface (both
  sampled), mean and maximum.
"""

import math
import random
import struct
import sys

TERMINAL = 3.0e38


# ---------------------------------------------------------------- reading


class Reader:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v

    def array(self, item_bytes):
        (count,) = self.take("Q")
        start = self.o
        self.o += count * item_bytes
        return count, start


def read_mlod(path):
    data = open(path, "rb").read()
    r = Reader(data)
    (endian,) = r.take("B")
    if endian != 1:
        raise SystemExit(path + ": big-endian files are not supported here")
    magic, version = r.take("II")
    if magic != 0x444F4C4D:
        raise SystemExit(path + ": not an .mlod file")
    r.take("iiiifff?")  # algorithm, max/min sizes, weights, optimize
    r.take("??fi")  # lod, lodSpatial, ratio, partition

    n, at = r.array(24)
    verts = [struct.unpack_from("<3f", data, at + 24 * i) for i in range(n)]
    if version >= 2:
        r.array(8)  # texcoords
    n, at = r.array(4)
    r.take("3f3f3fff")  # bounds, center, radius, avgEdgeLength

    n, at = r.array(16)
    meshlets = [struct.unpack_from("<4I", data, at + 16 * i) for i in range(n)]
    n, at = r.array(4)
    mverts = struct.unpack_from("<%dI" % n, data, at)
    n, at = r.array(1)
    mtris = data[at : at + n]
    n, at = r.array(32)
    bounds = [struct.unpack_from("<8f", data, at + 32 * i) for i in range(n)]
    n, at = r.array(48)
    lods = [struct.unpack_from("<10f2I", data, at + 48 * i) for i in range(n)]
    n, at = r.array(1)
    masks = data[at : at + n]

    clusters = []
    for i, (vo, to, vc, tc) in enumerate(meshlets):
        tris = []
        for t in range(tc):
            a, b, c = (mverts[vo + mtris[to + 3 * t + k]] for k in range(3))
            tris.append((a, b, c))
        lod = lods[i]
        clusters.append(
            {
                "level": lod[10],
                "terminal": lod[9] >= TERMINAL,
                "tris": tris,
                "verts": vc,
                "mask": masks[to // 3 : to // 3 + tc],
                "cutoff": bounds[i][7],
            }
        )
    return verts, clusters


def level_cut(clusters, k):
    return [c for c in clusters if c["level"] == k or (c["level"] < k and c["terminal"])]


# ---------------------------------------------------------------- geometry


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(a):
    return math.sqrt(dot(a, a))


def tri_area(p, q, s):
    return 0.5 * length(cross(sub(q, p), sub(s, p)))


def radius_ratio(p, q, s):
    e = dot(sub(q, p), sub(q, p)) + dot(sub(s, q), sub(s, q)) + dot(sub(p, s), sub(p, s))
    return 4.0 * math.sqrt(3.0) * tri_area(p, q, s) / e if e > 0 else 0.0


def closest_on_triangle(p, a, b, c):
    """Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        x = a
    else:
        bp = sub(p, b)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0 and d4 <= d3:
            x = b
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                v = d1 / (d1 - d3)
                x = (a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2])
            else:
                cp = sub(p, c)
                d5, d6 = dot(ab, cp), dot(ac, cp)
                if d6 >= 0 and d5 <= d6:
                    x = c
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0 and d2 >= 0 and d6 <= 0:
                        w = d2 / (d2 - d6)
                        x = (a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2])
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            x = (b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2]))
                        else:
                            denom = 1.0 / (va + vb + vc)
                            v, w = vb * denom, vc * denom
                            x = (a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w)
    d = sub(p, x)
    return dot(d, d)


class TriangleGrid:
    """Uniform grid over triangles' bounding boxes, for closest-point queries."""

    def __init__(self, verts, tris, cell):
        self.verts, self.tris, self.cell = verts, tris, cell
        self.cells = {}
        for i, t in enumerate(tris):
            ps = [verts[v] for v in t]
            lo = [int(math.floor(min(p[k] for p in ps) / cell)) for k in range(3)]
            hi = [int(math.floor(max(p[k] for p in ps) / cell)) for k in range(3)]
            for x in range(lo[0], hi[0] + 1):
                for y in range(lo[1], hi[1] + 1):
                    for z in range(lo[2], hi[2] + 1):
                        self.cells.setdefault((x, y, z), []).append(i)

    def distance(self, p):
        cx, cy, cz = (int(math.floor(p[k] / self.cell)) for k in range(3))
        best, seen = float("inf"), set()
        ring = 0
        while True:
            for x in range(cx - ring, cx + ring + 1):
                for y in range(cy - ring, cy + ring + 1):
                    for z in range(cz - ring, cz + ring + 1):
                        if max(abs(x - cx), abs(y - cy), abs(z - cz)) != ring:
                            continue
                        for i in self.cells.get((x, y, z), ()):
                            if i in seen:
                                continue
                            seen.add(i)
                            a, b, c = (self.verts[v] for v in self.tris[i])
                            best = min(best, closest_on_triangle(p, a, b, c))
            # every triangle closer than `ring` cells has been seen
            if best < float("inf") and math.sqrt(best) <= ring * self.cell:
                return math.sqrt(best)
            ring += 1
            if ring > 64:
                return math.sqrt(best)


# ---------------------------------------------------------------- statistics


def median(values):
    s = sorted(values)
    return s[len(s) // 2] if s else 0.0


def deviation(verts, cut_tris, base_tris, edge, samples, rng):
    """Two-sided vertex-sampled distance between a level and level 0, in level-0 edge lengths."""
    used = sorted({v for t in cut_tris for v in t})
    base_used = sorted({v for t in base_tris for v in t})
    to_base = TriangleGrid(verts, base_tris, 2.0 * edge)
    to_level = TriangleGrid(verts, cut_tris, 4.0 * edge)
    a = [to_base.distance(verts[v]) for v in rng.sample(used, min(samples, len(used)))]
    b = [to_level.distance(verts[v]) for v in rng.sample(base_used, min(samples, len(base_used)))]
    d = a + b
    return sum(d) / len(d) / edge, max(d) / edge


def analyse(path, samples, levels_wanted):
    verts, clusters = read_mlod(path)
    levels = max(c["level"] for c in clusters) + 1
    base = [t for c in level_cut(clusters, 0) for t in c["tris"]]
    edges = []
    for t in base:
        for i in range(3):
            edges.append(length(sub(verts[t[i]], verts[t[(i + 1) % 3]])))
    edge = sum(edges) / len(edges)
    rng = random.Random(1)

    rows = []
    for k in range(levels):
        if levels_wanted and k not in levels_wanted:
            continue
        cut = level_cut(clusters, k)
        tris = [t for c in cut for t in c["tris"]]
        q = [radius_ratio(*(verts[v] for v in t)) for t in tris]
        iq = []
        for c in cut:
            area = sum(tri_area(*(verts[v] for v in t)) for t in c["tris"])
            perimeter = 0.0
            for t, m in zip(c["tris"], c["mask"]):
                for i in range(3):  # bit i: edge opposite local vertex i
                    if m & (1 << i):
                        perimeter += length(sub(verts[t[(i + 1) % 3]], verts[t[(i + 2) % 3]]))
            if perimeter > 0:
                iq.append(4.0 * math.pi * area / (perimeter * perimeter))
        row = {
            "level": k,
            "clusters": len(cut),
            "triangles": len(tris),
            "tris/cluster": len(tris) / len(cut),
            "verts/cluster": sum(c["verts"] for c in cut) / len(cut),
            "vrefs/tri": sum(c["verts"] for c in cut) / len(tris),
            "q median": median(q),
            "slivers %": 100.0 * sum(1 for x in q if x < 0.3) / len(q),
            "cluster IQ": median(iq),
            "cullable %": 100.0 * sum(1 for c in cut if c["cutoff"] < 1.0) / len(cut),
        }
        if k > 0 and samples > 0:
            row["dev mean"], row["dev max"] = deviation(verts, tris, base, edge, samples, rng)
        rows.append(row)
        print("  level %d done" % k, file=sys.stderr)
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    samples = 4000
    levels = None
    for a in sys.argv[1:]:
        if a.startswith("--samples="):
            samples = int(a.split("=")[1])
        if a.startswith("--levels="):
            levels = {int(x) for x in a.split("=")[1].split(",")}
    for path in args:
        rows = analyse(path, samples, levels)
        print("\n" + path)
        keys = list(rows[-1].keys())
        print("| " + " | ".join(keys) + " |")
        print("|" + "---|" * len(keys))
        for row in rows:
            cells = []
            for key in keys:
                v = row.get(key, "")
                cells.append(("%.3g" % v if isinstance(v, float) else str(v)) if v != "" else "-")
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
