# meshlets

Interactive meshlet viewer: loads a PLY/OBJ model, splits it into meshlets with
[meshoptimizer](external/meshoptimizer), and renders them with Vulkan task + mesh
shaders (`VK_EXT_mesh_shader`), with a vertex-shader fallback for GPUs without them.
The UI is a Dear ImGui sidebar on the left (on GLFW); drag its right edge to resize it.

- Task shader: per-meshlet frustum and normal-cone (backface) culling
- Mesh shader: one workgroup per meshlet, up to 256 vertices / 256 triangles
- Meshlets colored with a gradient by LOD level (blue = full detail to orange = coarsest), or by
  triangle or normal, or Shaded with any of Blender's matcaps (thumbnail selector)
- Black outlines around each meshlet's perimeter plus a thinner triangle wireframe (on by
  default), drawn in the fragment shader with widths in model units: they scale with the mesh,
  not the screen, and are anti-aliased, fading out when thinner than a pixel
- Meshlet inspector: highlight one meshlet by index
- Meshlet builders: `buildMeshlets`, `buildMeshletsFlex`, `buildMeshletsSpatial`, `buildMeshletsScan`
- Cluster LOD (Nanite-style, on by default; `--no-lod` for a single meshlet level) with meshoptimizer's `demo/clusterlod.h`: a DAG of progressively
  simplified clusters. The task shader picks a crack-free cut per frame from a pixel error
  threshold, or shows a fixed level. Automatic selection walks a BVH over the cluster groups
  (`clodBuildHierarchy`) in compute passes, one per tree depth, so only candidate clusters reach
  the task shader (Lucy: identical cut, 3.0 -> 1.2 ms). You can freeze the LOD + culling camera
  to inspect the cut from elsewhere and see per-level cluster/triangle stats. Lucy: 19 levels,
  45 s single-threaded build; the full view draws ~85K of 28M triangles at 1 px
- Loading and meshlet building run on a background thread (Lucy, 28M triangles: about 5s + 12s)

## Setup

```bash
git submodule update --init --recursive
```

Everything except the Vulkan driver is built from `external/`. The Vulkan SDK is not
needed: volk loads `libvulkan.so.1` at runtime, and glslang (built from source)
compiles the GLSL shaders into SPIR-V that is embedded in the binary.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

## Run

```bash
./build/meshlets                       # loads assets/model/metasphere.ply
./build/meshlets path/to/model.ply     # or -i/--input PATH
./build/meshlets --no-lod -a flex --max-vertices 128 --max-triangles 128 --min-triangles 64
./build/meshlets -c shaded --matcap metal_carpaint --outline-width 0.3 --wire-width 0.05
./build/meshlets --lod-error 2          # coarser cut (or --lod-level N for a fixed level)
./build/meshlets --help                # all options
```

Mouse: left-drag to orbit, Shift+left-drag or middle-drag to pan, right-drag or the
wheel to zoom. You can also drop a `.ply`/`.obj` file onto the window to load it.

Batch use: `--frames N --screenshot out.ppm` renders N frames after loading,
saves the last frame and exits. A one-line meshlet summary is printed to stdout
after every build.

## Layout

| Path | Contents |
| --- | --- |
| `src/model.*` | PLY (ascii / binary LE / BE) + OBJ loading, meshlet building and stats |
| `src/vk_context.*` | Vulkan instance/device/swapchain, buffers (buffer device address) |
| `src/renderer.*` | Mesh-shader and fallback pipelines, per-frame stats and GPU timing |
| `src/matcap.*` | Loads the matcap PNGs from `assets/matcap/` (stb_image) |
| `src/main.cpp` | Command-line options, ImGui UI, camera, background jobs |
| `shaders/` | `meshlet.task`, `meshlet.mesh`, `meshlet.vert`, `meshlet.frag`, `lod_traverse.comp`, `lod_finalize.comp`, `common.glsl` |
| `external/` | Submodules: meshoptimizer, imgui, argparse, glfw, volk, Vulkan-Headers, glslang, stb, Eigen (vector math, header-only), liblava (not built) |

`assets/matcap/*.png` are Blender's matcaps (`datafiles/studiolights/matcap`, CC0, see
`assets/matcap/license.txt`). They were converted from Blender's EXRs as
diffuse + specular (the look of a white material), clipped to [0, 1] and stored as 8-bit sRGB.

The default model is `assets/model/metasphere.ply` (66K vertices, 132K triangles). For a
large stress test, the Stanford Lucy scan (14M vertices, 28M triangles, 533 MB) from the
Stanford 3D Scanning Repository (http://graphics.stanford.edu/data/3Dscanrep/) works well:
`./build/meshlets path/to/lucy.ply --z-up`. It is not in the repository; `assets/lucy.ply`
is git-ignored if you keep a local copy there.
Its terms allow research use and ask for attribution.
