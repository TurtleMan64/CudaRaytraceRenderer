# CUDA Raytrace Renderer

* A program that is capable of rendering a 3D model in realtime, without using traditional 3D rendering APIs (OpenGL, DirectX, etc.)
* Uses CUDA to cast each screen pixel as a ray in parallel on the graphics card. This allows the ~2 million ray casts per frame to happen in realtime.
* Supports up to ~330 triangles.
* Can define mirror type materials. Mirrors reflect incoming rays, as defined by a normal map.
* Uses SDL2 to render final image to screen.

## Example scene:
![Orca over ocean](https://github.com/TurtleMan64/CudaRaytraceRenderer/blob/main/Example.png?raw=true)
