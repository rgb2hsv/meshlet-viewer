---
layout: default
title: Building meshlets
description: The pipeline that turns a triangle mesh into meshlets and a cluster LOD DAG, with a close look at how the clustering works.
---

# Building meshlets

<p class="lede">How the viewer turns a triangle mesh into small, GPU-sized clusters of triangles (meshlets),
then into a hierarchy of progressively simplified clusters that the GPU can pick from every frame.</p>

<figure>
  <div class="grid grid-4">
    <figure><img src="assets/img/lod0.jpg" alt="Metasphere at full detail, 1364 meshlets"><figcaption>L0 · 132K triangles</figcaption></figure>
    <figure><img src="assets/img/lod2.jpg" alt="Metasphere at DAG level 2"><figcaption>L2 · 33K</figcaption></figure>
    <figure><img src="assets/img/lod4.jpg" alt="Metasphere at DAG level 4"><figcaption>L4 · 8.3K</figcaption></figure>
    <figure><img src="assets/img/lod6.jpg" alt="Metasphere at DAG level 6"><figcaption>L6 · 2.1K</figcaption></figure>
  </div>
  <figcaption>The bundled <code>metasphere.ply</code> at four levels of the cluster DAG (<code>--lod-level N</code>). Black lines are meshlet
  borders, thin lines are triangles, and the color encodes the level (blue = full detail, orange = coarsest). Every level is built from the same
  clustering steps described below.</figcaption>
</figure>

<div class="toc" markdown="1">
**Contents**

1. [What a meshlet is](#what-a-meshlet-is)
2. [Pipeline overview](#pipeline-overview)
3. [Preparing the mesh](#preparing-the-mesh)
4. [Clustering one level](#clustering-one-level)
5. [Per-meshlet data](#per-meshlet-data)
6. [Cluster LOD: building the DAG](#cluster-lod-building-the-dag)
7. [Picking a cut at runtime](#picking-a-cut-at-runtime)
8. [Parameters](#parameters)
</div>

## What a meshlet is

Mesh shaders replace the fixed vertex fetch and primitive assembly stages with compute-like workgroups. Each
workgroup emits a small, self-contained piece of geometry: up to a few hundred vertices and triangles. A
**meshlet** is one such piece, prepared offline. The viewer allows up to 256 vertices and 256 triangles per
meshlet (the defaults are 64 and 124).

A meshlet is stored as two arrays plus a small header (`meshopt_Meshlet`):

<figure>
  <img src="assets/fig/layout.svg" alt="Meshlet header pointing into a vertex index array and a byte triangle array">
  <figcaption>Each meshlet references a run of <code>vertices[]</code> (32-bit indices into the mesh's vertex buffer) and a run of
  <code>triangles[]</code> (three 8-bit indices into that run). The 8-bit local indices are why a meshlet can hold at most 256
  vertices.</figcaption>
</figure>

The mesh shader loads the meshlet's vertices once into its output arrays, then writes its triangles using the
local indices, so a vertex shared by several triangles in the meshlet is transformed only once. A good clustering
therefore wants:

- **Few vertices per triangle.** A compact, roughly round patch shares most of its vertices between triangles; a
  long strip doesn't. On a regular mesh, a disk of \(V\) vertices holds about \(2V\) triangles minus its boundary length, so 64
  vertices hold roughly 100 triangles. That's why the metasphere's meshlets average 64 vertices and 97 triangles: they hit the vertex limit first.
- **Full meshlets.** Every meshlet costs a workgroup, so half-empty ones waste GPU time.
- **Spatially tight meshlets.** Bounding spheres and normal cones are used to cull whole meshlets, which only works
  when each one is small and roughly flat.

## Pipeline overview

<figure>
  <img src="assets/fig/pipeline.svg" alt="Pipeline: load, weld, vertex cache order, clusterize; then a loop of partition, lock borders, simplify, re-clusterize; then per-meshlet data and the group BVH; then GPU traversal, task, mesh and fragment shaders">
  <figcaption>The build runs once per model on a background thread (<code>loadModel</code> and <code>buildMeshlets</code> in
  <code>src/model.cpp</code>). The green loop is meshoptimizer's <code>clodBuild</code> (<code>external/meshoptimizer/demo/clusterlod.h</code>);
  with <code>--no-lod</code> it is skipped and a single level is built with one of the four builders.</figcaption>
</figure>

## Preparing the mesh

`loadModel` reads a PLY or OBJ file into a flat vertex buffer (position + normal) and a triangle index buffer, then:

1. **Welds** identical vertices (`meshopt_generateVertexRemap` for OBJ, whose faces index positions and normals
   separately). Clustering works on shared vertices, so an unwelded mesh would look like a soup of disconnected triangles.
2. **Computes normals** when the file has none, by summing area-weighted face normals at each vertex.
3. **Reorders the triangles for vertex-cache locality** (`meshopt_optimizeVertexCache`, skip with `--no-vertex-cache`).
   Triangles that share vertices end up close together in the index buffer. The scan builder depends on this directly,
   and it makes the greedy builder's fallback search cheaper.

## Clustering one level

All builders share one primitive, `appendMeshlet`: add a triangle to the current meshlet, assigning local indices to any
of its vertices the meshlet hasn't seen yet. If that would exceed the vertex or triangle limit, the current meshlet is
written out first and a new one started. A per-vertex `used[]` array maps global vertex → local index (or −1) so this
check is O(1). The builders differ only in **which triangle comes next**.

<figure>
  <img src="assets/fig/builders.svg" alt="The same 2D grid clustered three ways: horizontal stripes from scan, irregular round patches from greedy, rectangular tiles from spatial">
  <figcaption>The three strategies on a 352-triangle test grid with a limit of 16 vertices and 18 triangles. These panels come from
  small Python ports of the meshoptimizer algorithms (<code>docs/_figures/gen_figures.py</code>). The grid's index order is row by row,
  which is the worst case for scan. On real meshes after vertex-cache ordering, scan does much better, but its meshlets still have
  ragged borders.</figcaption>
</figure>

### Scan (`-a scan`)

`meshopt_buildMeshletsScan` walks the index buffer in order and cuts a meshlet whenever the next triangle doesn't fit.
It's linear-time and needs no positions, but the result is only as good as the input order: it never looks at
neighbors, so meshlets follow whatever path the index buffer happens to take.

### Greedy growth (`-a greedy`, `-a flex`, and the LOD builder)

`meshopt_buildMeshlets` is `meshopt_buildMeshletsFlex` with `min_triangles = max_triangles` and no splitting. It grows each
meshlet one triangle at a time from a seed, always preferring triangles that touch the meshlet already. This is the
builder `clodBuild` uses for every level of the LOD hierarchy (unless `--lod-spatial`).

**Setup.** The builder computes:

- **Triangle adjacency**: for each vertex, the list of triangles that use it.
- **Live counts**: for each vertex, how many of its triangles haven't been placed yet. Whenever a triangle is emitted it's
  removed from its vertices' adjacency lists, so these counts drop as the mesh is used up.
- **Per-triangle centroid and unit normal.**
- **Expected meshlet radius**, assuming a meshlet is a square patch of `max_triangles` average-sized triangles:

$$ r_{\text{exp}} = \tfrac{1}{2}\sqrt{\bar{A}\cdot T_{\max}} $$

where \\(\bar{A}\\) is the average triangle area (half the cross-product length).

- **A kd-tree over the triangle centroids**, used when growth gets stuck.

The first seed is the triangle whose centroid is closest to the minimum corner of the mesh's bounding box. Starting at a
corner makes meshlets sweep across the surface instead of leaving holes in the middle.

**Choosing the next triangle.** For every vertex already in the meshlet, every live triangle around it is a candidate.
Each candidate first gets a **topological priority** from the number of new vertices it would add (`extra`) and the live
counts of its vertices:

| priority | condition | why |
|---|---|---|
| 0 | `extra == 0`: all three vertices are already in the meshlet | a free triangle; it costs no vertex slot |
| 1 | one of its vertices has a live count of 1 | otherwise this triangle would be left dangling, and would cost a whole new vertex set later |
| 1 + extra | two of its vertices have a live count of 2 | taking it leaves a neighbor dangling, which the next step can pick up as priority 1 |
| 2 + extra | anything else | 3 or 4, by vertex cost |

Lower priority always wins. Ties are broken by a **geometric score** (lower is better) combining distance to the
meshlet's running centroid \\(c\\) and agreement with its average normal \\(n\\):

$$ \text{score} = \left(1 + \frac{\lVert p_t - c\rVert}{r_{\text{exp}}}\,(1 - w)\right)\cdot \max\!\left(10^{-3},\; 1 - w\,(n_t\cdot n)\right) $$

\\(w\\) is the **cone weight** (`--cone-weight`, default 0.25). At \\(w = 0\\) the score is pure distance and meshlets come out round;
as \\(w\\) grows, triangles that face the same way as the meshlet are preferred. That narrows each meshlet's normal cone, so more
meshlets can be backface culled as a whole, at the cost of less compact shapes. (`clodBuild` passes \\(w = 0\\): after simplification
cone culling matters less than compactness.)

<figure>
  <img src="assets/fig/greedy-step.svg" alt="A meshlet in the middle of growing, with neighboring triangles labeled by priority 1 to 4 and the chosen triangle outlined in red">
  <figcaption>One step of greedy growth. The current meshlet (blue) has 9 triangles and 10 vertices. Each candidate around it is labeled
  with its priority. The one outlined in red is priority 1: it uses a vertex whose only remaining triangle it is. Ties within a priority
  go to the candidate closest to the meshlet's centroid (red circle).</figcaption>
</figure>

**When the meshlet is full.** If the best candidate would exceed `max_vertices` or `max_triangles`, the meshlet is closed and a
new one needs a seed. Instead of continuing from the rejected candidate, the builder keeps a list of up to 256 **seed
triangles**. For each vertex of the meshlet that just closed, the adjacent triangle with the smallest total live count is
added (up to 4 per meshlet). The next seed is the one with the lowest live count, closest to the starting corner. Low live
counts mean triangles at the edge of the unprocessed region, so the next meshlet starts right where the last one left off
and the front of finished meshlets advances as one piece.

**When no neighbor is left.** If no triangle touches the meshlet (an island, or a region already surrounded by finished meshlets),
the kd-tree supplies the nearest unplaced triangle to the meshlet's centroid. With `--split-factor F > 0` (flex only), if that triangle
is farther than \\(F \cdot r_{\text{exp}}\\) away and the meshlet already has `--min-triangles`, the meshlet is closed early rather than
stretched across a gap. This is how flex trades a few smaller meshlets for tighter bounds.

The whole loop is \\(O(T \cdot k)\\) for \\(T\\) triangles and average valence \\(k\\), since each step only scans the adjacency of the
current meshlet's vertices.

### Spatial / SAH (`-a spatial`, `--lod-spatial`)

`meshopt_buildMeshletsSpatial` ignores connectivity and partitions the triangles top-down, like building a BVH for ray tracing:

1. Compute each triangle's bounding box and centroid, and **radix-sort** the triangle list three times, once by centroid along each of x, y and z.
2. **Split recursively.** If the current range fits in one meshlet (≤ `max_triangles`, ≤ `max_vertices` unique vertices), it becomes a leaf.
   Otherwise, for each axis, sweep the sorted order and evaluate every cut position \\(i\\) with the **surface area heuristic**:

   $$ \text{cost}(i) = A(\text{left}_i)\cdot|\text{left}_i| + A(\text{right}_i)\cdot|\text{right}_i| \;+\; f\cdot\big(\text{waste}_L\,A_L + \text{waste}_R\,A_R\big) $$

   \\(A\\) is the surface area of the side's bounding box, computed for all cuts in two linear passes (prefix from the left, suffix from the right).
   The second term, weighted by `--fill-weight` \\(f\\), charges for slots a side would leave empty once rounded up to whole meshlets, so
   the splitter favors cuts that produce full meshlets.
3. **Only cuts that can be tiled are allowed.** A side with \\(n\\) triangles is admissible only if \\(n\\) can be written as a sum of sizes in
   `[min_triangles, max_triangles]`. Otherwise the recursion could end up with a leftover piece too small to be a valid meshlet.
4. Take the cheapest cut over all three axes, partition the other two sorted orders stably, and recurse on both halves.

<figure>
  <img src="assets/fig/sah.svg" alt="The test grid split into a left and right half, and SAH cost curves for x and y splits with a minimum near the middle of the x curve">
  <figcaption>The first SAH split of the test grid. The cost is evaluated at every admissible cut along both axes. The cheapest is the vertical
  cut near the middle of the x sweep (circled), since it halves the widest dimension.</figcaption>
</figure>

The result is tiled, very full meshlets with tight boxes, which is good for ray tracing and culling. But it doesn't optimize for shared vertices, so
vertex counts per meshlet are higher than with greedy growth.

<figure>
  <div class="grid grid-3">
    <figure><img src="assets/img/algo-scan.jpg" alt="Scan meshlets on the metasphere"><figcaption>scan · 1001 meshlets</figcaption></figure>
    <figure><img src="assets/img/algo-greedy.jpg" alt="Greedy meshlets on the metasphere"><figcaption>greedy · 869</figcaption></figure>
    <figure><img src="assets/img/algo-spatial.jpg" alt="Spatial meshlets on the metasphere"><figcaption>spatial · 1335</figcaption></figure>
  </div>
  <figcaption>The three builders on the metasphere with <code>--no-lod</code> and default limits (64 vertices / 124 triangles; spatial with 32 minimum
  triangles). Counts are the meshlets visible from this view after culling.</figcaption>
</figure>

### Optimizing each meshlet

Without LOD, every meshlet is then passed to `meshopt_optimizeMeshlet`, which reorders its triangles and local vertices for locality
inside the meshlet. `clodBuild` does the same internally (`optimize_clusters`). Turn it off with `--no-optimize-meshlets`.

## Per-meshlet data

After clustering, `buildMeshlets` computes three more things for every meshlet, in parallel:

- **Bounding sphere and normal cone** (`meshopt_computeMeshletBounds`). The cone axis is the average triangle normal, and the cutoff
  is derived from the widest angle between that axis and any triangle normal. The task shader rejects a meshlet whose every triangle
  faces away from the camera at \\(e\\):

  $$ (c - e)\cdot a \;\ge\; \text{cutoff}\cdot\lVert c - e\rVert + r $$

  using the sphere \\((c, r)\\) and cone axis \\(a\\). The \\(+r\\) term makes the test valid anywhere on the meshlet, not just at its center.
- **Perimeter edge masks.** For each triangle, a 3-bit mask of which edges lie on the meshlet's border: an edge used by exactly one
  triangle of the meshlet. The fragment shader draws the black meshlet outlines from these bits.
- **Statistics**: fill rates, average cone angle, and the fraction of meshlets with a usable cone, shown in the sidebar.

## Cluster LOD: building the DAG

With LOD on (the default), the viewer calls `clodBuild`, meshoptimizer's implementation of the Nanite-style cluster hierarchy.
The goal is a set of clusters at many levels of detail, such that the GPU can draw any **mix** of levels, fine near the camera
and coarse far away, without cracks between neighbors at different levels.

### Why simplify groups, not clusters

Simplifying each cluster on its own would mean locking its whole border so that it still matches its neighbors. After a few
levels the borders would hold most of the triangles and simplification would stall. The fix (from Karis' *Nanite* talk) is to
alternate which edges are locked:

1. **Group** neighboring clusters, about 16 at a time.
2. **Lock** only the edges between groups, then **simplify** each group as one mesh to about half its triangles. Edges *inside* the group, which
   were cluster borders before, are now free to collapse.
3. **Re-clusterize** the simplified group into new clusters (roughly half as many).
4. Repeat on the new clusters. The next level's groups are different, so the edges locked this time are interior next time.

Because each group's simplified clusters share the same locked border as the group's original clusters, the GPU can swap a whole group
between its fine and coarse versions without a crack.

### Partitioning clusters into groups

`meshopt_partitionClusters` is a bottom-up (agglomerative) clustering of clusters, based on Kurita's heap-based method:

1. Build a **cluster adjacency graph**. Two clusters are adjacent if they share a vertex. The edge weight is the number of shared vertices.
   Vertices are first remapped by position (`meshopt_generatePositionRemap`) so clusters separated by a normal or UV seam still count as connected.
2. Start with one group per cluster, in a min-heap keyed on vertex count.
3. Pop the **smallest** group. If it has reached the target size, emit it. Otherwise merge it with the adjacent group that maximizes

   $$ \text{score} = s \cdot\left(\frac{1}{\sqrt{v_a}} + \frac{1}{\sqrt{v_b}}\right)\cdot\big(1 + 0.4\,\beta\big) $$

   where \\(s\\) is the number of shared vertices, \\(v_a, v_b\\) are the groups' vertex counts, and \\(\beta \in [0, 1]\\) is a bounding-sphere
   overlap term that favors spatially close groups. Dividing by \\(\sqrt{v}\\) normalizes by the expected border length of each group, so
   the merge that hides the most border relative to size wins. Merges that would exceed 4/3 of the target size are skipped.
4. Push the merged group back and repeat.
5. Finally, small leftover groups are merged by spatial proximity alone (a kd-split over group centers), which catches disconnected pieces.

Popping the smallest group first keeps group sizes even. Maximizing shared vertices keeps each group compact, with a short border, which is exactly the set of edges
that will be locked.

<figure>
  <img src="assets/fig/partition.svg" alt="Left: clusters with an adjacency graph drawn between their centers. Right: the clusters merged into five groups, with the vertices on group borders marked red">
  <figcaption>Left: greedy clusters of the test grid and their adjacency graph (thicker = more shared vertices). Right: the result of
  partitioning with a target of 4 clusters per group (the viewer uses 16, <code>--lod-partition</code>). Dashed lines are the old cluster
  borders, now free to simplify. The red vertices on group borders are locked.</figcaption>
</figure>

### Locking, simplifying and error tracking

For each level, `clodBuild`:

- **Locks group borders.** Every vertex used by more than one group gets a lock bit, and so does every vertex at the same
  position, so seams stay consistent.
- **Simplifies each group** with `meshopt_simplifyWithAttributes` to `--lod-ratio` × its triangle count (default 0.5). It runs with:
  - normals as attributes (weight 0.5 each), so shading creases are preserved;
  - *permissive* mode, which allows collapses across attribute seams where that does little visible damage;
  - *sparse* mode, which only touches the group's vertices;
  - an *absolute* error result in model units.

  If the target isn't reached, it falls back to `meshopt_simplifySloppy` (vertex clustering, ignores topology) and multiplies the reported error by 2 to account for the
  rougher result.
- **Marks stuck groups terminal.** If simplification keeps more than 85% of the triangles, the group is marked terminal (`error = FLT_MAX`)
  and not simplified further.
- **Keeps errors monotonic.** A group's bounds are the merged spheres of its clusters' bounds, not the tight bounds of the simplified mesh.
  Its error is

  $$ e_{\text{group}} = \max\big(\max_{c \in \text{group}} e_c,\; e_{\text{simplify}}\big) $$

  so a coarser level never reports a *smaller* error than the level below it, and its bounding sphere always contains its children's.
  Monotonicity is what lets the runtime make each decision locally (next section).
- **Re-clusterizes** the simplified triangles with the same greedy builder (`--min-triangles` / `--max-triangles`, split factor 2) and gives the new
  clusters the group's bounds and error.

The loop stops when one cluster is left, or when every group is terminal.

Each group is reported to the viewer through a callback. For every cluster the viewer stores two error spheres (`MeshletLod` in `src/model.h`):

- **self**: the error of the group that *produced* this cluster (0 for original geometry), i.e. how wrong this cluster is.
- **coarse**: the error of the group this cluster *belongs to*, i.e. how wrong its replacement would be.

<figure>
  <img src="assets/fig/dag.svg" alt="A four-level DAG: sixteen L0 clusters in four groups, each group producing two L1 clusters, and so on. Some clusters at L0, L1 and L2 are filled to show the selected cut">
  <figcaption>The cluster DAG. Dashed boxes are groups; each group of four is simplified into two clusters on the level above. Group errors are
  projected to pixels for one camera position. Red groups are above the 1 px threshold, green ones are under it. The filled clusters form a valid
  cut: they cover the surface exactly once, mixing L0, L1 and L2.</figcaption>
</figure>

On the metasphere (132K triangles) this takes 0.28 s and produces 12 levels and 2933 clusters in total: 1364 at level 0, with the triangle
count halving at every level (132K, 66K, 33K, … 128, 64). The Lucy statue (28M triangles) reaches 19 levels in about 45 s single-threaded.

## Picking a cut at runtime

A cluster's object-space error \\(e\\) is projected to pixels using the closest point of its error sphere \\((c, r)\\) to the camera:

$$ \text{px}(e, c, r) = \frac{e}{\max(\lVert c - \text{eye}\rVert - r,\; z_{\text{near}})}\cdot \frac{\cot(\text{fov}_y/2)}{2}\cdot H $$

where \\(H\\) is the viewport height. A cluster is drawn exactly when

$$ \text{px}(\text{coarse}) > \tau \quad\text{and}\quad \text{px}(\text{self}) \le \tau $$

for the threshold \\(\tau\\) (`--lod-error`, default 1 px). In words: the cluster's replacement would be visibly wrong, but the cluster itself isn't.
Since errors and spheres grow monotonically up the DAG, both quantities change in step along any path from a leaf to the root, so exactly one
cluster on each path passes. That's why the selected clusters tile the surface with no gaps or overlaps, without any communication between clusters.
And since all clusters of a group share the same *coarse* sphere and error, a group always switches as a unit, along its locked border.

Testing all clusters every frame would be wasteful, since most clusters of a large mesh are far too fine or too coarse. So the build also runs
`clodBuildHierarchy`, which builds an 8-wide **BVH over the groups**, one tree per DAG level. Each node stores the merged sphere and maximum error of its subtree.
On the GPU, `lod_traverse.comp` walks these trees breadth-first, one compute dispatch per tree depth:

- A node outside the frustum is skipped.
- A node whose projected error is ≤ \\(\tau\\) is skipped, because every group below it is fine enough to be replaced by its parent.
  Internal nodes use a small safety margin against float rounding in the merged bounds.
- A leaf (one group) emits the group's clusters whose own `self` error is ≤ \\(\tau\\) as **candidates**.

The task shader (`meshlet.task`) then does frustum and normal-cone culling on each candidate and launches one mesh-shader workgroup per survivor.
On Lucy, this cuts LOD selection from 3.0 ms to 1.2 ms for an identical cut.

<figure>
  <img src="assets/img/lod-auto.jpg" alt="The metasphere rendered with automatic LOD at a 4 pixel threshold, mixing cluster levels">
  <figcaption>Automatic selection with <code>--lod-error 4</code>: 15K triangles in 183 clusters, out of 132K at full detail. Regions that are flatter or farther
  away use coarser clusters, and borders between levels have no cracks because they follow locked group borders.</figcaption>
</figure>

## Parameters

| flag | default | effect |
|---|---|---|
| `--max-vertices N` | 64 | vertex limit per meshlet (≤ 256, from the 8-bit local indices) |
| `--max-triangles N` | 124 | triangle limit per meshlet |
| `--min-triangles N` | 32 | smallest meshlet flex and spatial may close early (also used by LOD) |
| `-a, --algorithm` | greedy | single-level builder with `--no-lod`: `greedy`, `flex`, `spatial`, `scan` |
| `--cone-weight W` | 0.25 | greedy/flex: trade compactness for narrower normal cones |
| `--split-factor F` | 2 | flex and LOD: close a meshlet instead of jumping farther than F × expected radius |
| `--fill-weight W` | 0.5 | spatial: prefer full meshlets over SAH quality |
| `--no-lod` | off | build one meshlet level instead of the DAG |
| `--lod-spatial` | off | LOD clusters with the SAH builder instead of flex |
| `--lod-ratio R` | 0.5 | triangle ratio each simplification step aims for |
| `--lod-partition N` | 16 | target clusters per group |
| `--lod-error PX` | 1 | runtime projected-error threshold |
| `--lod-level N` | auto | draw a fixed DAG level instead |

### Reproducing the figures

The diagrams in this page are generated by

```bash
python docs/_figures/gen_figures.py
```

and the renders by the viewer itself, e.g.

```bash
./build/meshlets --no-ui --width 900 --height 900 --background 1 1 1 --lod-level 2 --screenshot lod2.png
```

### Further reading

- Arseny Kapoulkine, [meshoptimizer](https://github.com/zeux/meshoptimizer): `src/clusterizer.cpp`, `src/partition.cpp`, `demo/clusterlod.h`.
- Brian Karis, Rune Stubbe, Graham Wihlidal, *Nanite: A Deep Dive*, SIGGRAPH 2021 Advances in Real-Time Rendering.
- Takio Kurita, *An efficient agglomerative clustering algorithm using a heap*, Pattern Recognition, 1991.
