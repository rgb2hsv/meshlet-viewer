# meshlets

Interactive meshlet viewer: loads a PLY/OBJ model, splits it into meshlets with
[meshoptimizer](external/meshoptimizer), and renders them with Vulkan task + mesh
shaders (`VK_EXT_mesh_shader`), with a vertex-shader fallback for GPUs without them.
The UI is Dear ImGui on GLFW.

- Task shader: per-meshlet frustum and normal-cone (backface) culling
- Mesh shader: one workgroup per meshlet, up to 256 vertices / 256 triangles
- Color by meshlet, triangle or normal, or Shaded with any of Blender's matcaps (thumbnail selector)
- Wireframe drawn in the fragment shader with a line width in model units (it scales with
  the mesh, not the screen; anti-aliased, fading out when thinner than a pixel)
- Meshlet inspector: highlight one meshlet by index
- Meshlet builders: `buildMeshlets`, `buildMeshletsFlex`, `buildMeshletsSpatial`, `buildMeshletsScan`
- Loading and meshlet building run on a background thread (Lucy, 28M triangles: about 5s + 12s)

## Setup

```bash
git submodule update --init --recursive
git lfs pull   # assets/lucy.ply (533 MB) is stored with Git LFS
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
./build/meshlets                       # loads assets/lucy.ply (Z-up)
./build/meshlets path/to/model.ply     # or -i/--input PATH
./build/meshlets -a flex --max-vertices 128 --max-triangles 128 --min-triangles 64
./build/meshlets -c shaded --matcap metal_carpaint --wireframe --wire-width 0.05
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
| `shaders/` | `meshlet.task`, `meshlet.mesh`, `meshlet.vert`, `meshlet.frag`, `common.glsl` |
| `external/` | Submodules: meshoptimizer, imgui, argparse, glfw, volk, Vulkan-Headers, glslang, stb, liblava (not built) |

`assets/matcap/*.png` are Blender's matcaps (`datafiles/studiolights/matcap`, CC0, see
`assets/matcap/license.txt`). They were converted from Blender's EXRs as
diffuse + specular (the look of a white material), clipped to [0, 1] and stored as 8-bit sRGB.

`assets/lucy.ply` is the Stanford Lucy scan from the Stanford 3D Scanning Repository.
Its terms allow research use and ask for attribution.
