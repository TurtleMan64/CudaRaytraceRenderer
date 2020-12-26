#include <SDL2/SDL.h> 
#include <SDL2/SDL_image.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstring>

#include "intellifix.h"

#include <ctime>
#include <ratio>
#include <chrono>

#include <Windows.h>

#include "toolbox/vector.h"
#include "toolbox/maths.h"
#include "collision/collisionmodel.h"
#include "collision/objloader.h"

#define BLOCKS 60
#define THREADS_PER_BLOCK (640)
//1920 9600/15
SDL_Window* window = nullptr;

SDL_Surface* windowSurface = nullptr;
int* displayBufferGPU = nullptr;
#define displayWidth 1600
#define displayHeight 900

//temporary hard coded texture
SDL_Surface* textureImg = nullptr;
int* textureBufferGPU = nullptr;
#define textureWidth 600
#define textureHeight 600

//cam stuff
Vector3d camPosition;
float camYaw = 91.0f;  //in degrees
float camPitch = -1.0f; //in degrees

float* camValuesGPU = nullptr; //position, yaw, pitch
#define fovV 60.0f
#define fovH 91.492844f
#define nearPlane 0.15f

//list of triangles data struct
__constant__ char trianglesGPU[32000];

cudaError_t setupGlobalVars();

cudaError_t render();

__device__ inline void vec3normalize(float vec3[])
{
    float len = sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2]);
    vec3[0]/=len;
    vec3[1]/=len;
    vec3[2]/=len;
}

__device__ inline float vec3len(const float a[])
{
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

__device__ inline float vec3dot(const float a[], const float b[])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ inline void vec3cross(float out[], const float a[], const float b[])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline float copysignGPU(const float val)
{
    int asInt;
    memcpy(&asInt, &val, 4);
    asInt = (asInt >> 31) & 0x1;
    return (2*(float)asInt)-1;
    //if (val >= 0.0)
    //{
    //    return 1.0;
    //}
    //else
    //{
    //    return -1.0;
    //}
}

__device__ void interpolateUvTriangle(float out[],
                                      const float p1x, const float p1y, const float p1z,
                                      const float p2x, const float p2y, const float p2z,
                                      const float p3x, const float p3y, const float p3z,
                                      const float uv1x, const float uv1y,
                                      const float uv2x, const float uv2y,
                                      const float uv3x, const float uv3y,
                                      const float fx, const float fy, const float fz)
{
    float f[] = {fx, fy, fz};
    float p1[] = {p1x, p1y, p1z};
    float p2[] = {p2x, p2y, p2z};
    float p3[] = {p3x, p3y, p3z};

    // calculate vectors from point f to vertices p1, p2 and p3:
    float f1[] = {p1[0]-f[0], p1[1]-f[1], p1[2]-f[2]};
    float f2[] = {p2[0]-f[0], p2[1]-f[1], p2[2]-f[2]};
    float f3[] = {p3[0]-f[0], p3[1]-f[1], p3[2]-f[2]};

    // calculate the areas (parameters order is essential in this case):
    float va[]  = {0,0,0};
    float va1[] = {0,0,0};
    float va2[] = {0,0,0};
    float va3[] = {0,0,0};
    float p1mp2[] = {p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]};
    float p1mp3[] = {p1[0]-p3[0], p1[1]-p3[1], p1[2]-p3[2]};
    vec3cross(va, p1mp2, p1mp3);  // main triangle cross product
    vec3cross(va1, f2, f3);  // p1's triangle cross product
    vec3cross(va2, f3, f1);  // p2's triangle cross product
    vec3cross(va3, f1, f2); // p3's triangle cross product

    float a = vec3len(va); // main triangle area
    // calculate barycentric coordinates with sign:
    float a1 = (vec3len(va1)/a) * copysignGPU(vec3dot(va, va1));
    float a2 = (vec3len(va2)/a) * copysignGPU(vec3dot(va, va2));
    float a3 = (vec3len(va3)/a) * copysignGPU(vec3dot(va, va3));

    out[0] =   uv1x*a1 + uv2x*a2 + uv3x*a3;
    out[1] = -(uv1y*a1 + uv2y*a2 + uv3y*a3);
}

__device__ bool checkPointInTriangle2DGPU(
	const float x,  const float y,
	const float x1, const float y1,
	const float x2, const float y2,
	const float x3, const float y3);

__device__ bool checkPointInTriangle3DGPU(
	const float checkx, const float checky, const float checkz,
	const int triIdx);

__device__ int inline pixelFromRgbGPU(const float r, const float g, const float b);

__device__  void rotatePointGPU(float result[],
    const float point[],
	const float rotationAxis[],
	const float theta);

__device__ void calcWorldSpaceDirectionVectorFromScreenSpaceCoordsGPU(float out[], const float clickPosX, const float clickPosY, const float camYaw, const float camPitch);

__device__ float inline toRadiansGPU(const float deg);

__device__ int sampleTriangleColorGPU(const float cx, const float cy, const float cz, const int triIdx, const int* inTexture)
{
    int base = triIdx*93;
    float t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z, uv1x, uv1y, uv2x, uv2y, uv3x, uv3y;
    memcpy(&t1x, &trianglesGPU[ 4+base], 4);
    memcpy(&t1y, &trianglesGPU[ 8+base], 4);
    memcpy(&t1z, &trianglesGPU[12+base], 4);
    memcpy(&t2x, &trianglesGPU[16+base], 4);
    memcpy(&t2y, &trianglesGPU[20+base], 4);
    memcpy(&t2z, &trianglesGPU[24+base], 4);
    memcpy(&t3x, &trianglesGPU[28+base], 4);
    memcpy(&t3y, &trianglesGPU[32+base], 4);
    memcpy(&t3z, &trianglesGPU[36+base], 4);

    memcpy(&uv1x, &trianglesGPU[72+base], 4); //uv
    memcpy(&uv1y, &trianglesGPU[76+base], 4);
    memcpy(&uv2x, &trianglesGPU[80+base], 4);
    memcpy(&uv2y, &trianglesGPU[84+base], 4);
    memcpy(&uv3x, &trianglesGPU[88+base], 4);
    memcpy(&uv3y, &trianglesGPU[92+base], 4);

    float uv[] = {0,0};
    interpolateUvTriangle(uv,
        t1x, t1y, t1z,
        t2x, t2y, t2z,
        t3x, t3y, t3z,
        uv1x, uv1y,
        uv2x, uv2y,
        uv3x, uv3y,
        cx, cy, cz);

    int x = (int)(textureWidth*uv[0]);
    int y = (int)(textureHeight*uv[1]);
    x = x % textureWidth;
    y = y % textureHeight;
    while (x < 0)
    {
        x = x + textureWidth;
    }

    while (y < 0)
    {
        y = y + textureHeight;
    }
    return inTexture[y*textureWidth + x];
}

__device__ bool checkCollisionGPU(char* result,
                                  const float px1, const float py1, const float pz1,
                                  const float px2, const float py2, const float pz2)
{
    //return 0;
    int numTriangles = 0;
    memcpy(&numTriangles, trianglesGPU, 4);

    float minDist = 100000000000.0f;
    int triangleId = -1;
    float cPos[] = {0,0,0};

    for (int i = 0; i < numTriangles; i++)
    {
        int base = i*93;

        //float A = *((float*)&trianglesGPU[40+base]); //this does not work
        //float B = *((float*)&trianglesGPU[44+base]);
        //float C = *((float*)&trianglesGPU[48+base]);
        //float D = *((float*)&trianglesGPU[52+base]);

        float A, B, C, D;
        memcpy(&A, &trianglesGPU[40+base], 4); //abcd
        memcpy(&B, &trianglesGPU[44+base], 4);
        memcpy(&C, &trianglesGPU[48+base], 4);
        memcpy(&D, &trianglesGPU[52+base], 4);

        //printf("alphabet = %f %f %f %f\n", A, B, C, D);

        float numerator = (A*px1 + B*py1 + C*pz1 + D);
		float denominator = (A*(px1 - px2) + B*(py1 - py2) + C*(pz1 - pz2));

        if (denominator != 0)
		{
			float u = (numerator / denominator);
			float cix = px1 + u*(px2 - px1);
			float ciy = py1 + u*(py2 - py1);
			float ciz = pz1 + u*(pz2 - pz1);
            //printf("u = %f\n", u);
            //printf("c = %f %f %f\n", cix, ciy, ciz);

            bool firstAbove;
            bool secondAbove;

			if (B != 0)
			{
				float planey1 = (((-A*px1) + (-C*pz1) - D) / B);
				float planey2 = (((-A*px2) + (-C*pz2) - D) / B);
				firstAbove = signbit(py1 - planey1);
				secondAbove = signbit(py2 - planey2);
			}
			else if (A != 0)
			{
				float planex1 = (((-B*py1) + (-C*pz1) - D) / A);
				float planex2 = (((-B*py2) + (-C*pz2) - D) / A);
				firstAbove = signbit(px1 - planex1);
				secondAbove = signbit(px2 - planex2);
			}
			else if (C != 0)
			{
				float planez1 = (((-B*py1) + (-A*px1) - D) / C);
				float planez2 = (((-B*py2) + (-A*px2) - D) / C);
				firstAbove = signbit(pz1 - planez1);
				secondAbove = signbit(pz2 - planez2);
			}

			if (secondAbove != firstAbove && checkPointInTriangle3DGPU(cix, ciy, ciz, i))
			{
				//what is the distance to the triangle? set it to maxdist
				float thisDist = (sqrt(fabs((cix - px1)*(cix - px1) + (ciy - py1)*(ciy - py1) + (ciz - pz1)*(ciz - pz1))));
				if (thisDist < minDist)
				{
                    //triangleCollide = true;
					//collideTriangle = currTriangle;
					cPos[0] = cix;
					cPos[1] = ciy;
					cPos[2] = ciz;
					minDist = thisDist;
                    triangleId = i;
					//finalModel = cm;
                    //return 1;
				}
			}
		}
    }

    if (triangleId >= 0)
    {
        memcpy(&result[ 0], &triangleId, 4);
        memcpy(&result[ 4], &cPos[0], 4);
        memcpy(&result[ 8], &cPos[1], 4);
        memcpy(&result[12], &cPos[2], 4);
        return true;
    }

    return false;
}

__device__ int doPixelGPU(const float* camValues, const int screenX, const int screenY, const int* inTexture)
{
    float worldCastDirection[] = {0, 0, 0};
    calcWorldSpaceDirectionVectorFromScreenSpaceCoordsGPU(worldCastDirection, (float)screenX, (float)screenY, camValues[3], camValues[4]);
    worldCastDirection[0] *= 1000;
    worldCastDirection[1] *= 1000;
    worldCastDirection[2] *= 1000;

    float gazePosition[] = {camValues[0], camValues[1], camValues[2]};
    gazePosition[0] += worldCastDirection[0];
    gazePosition[1] += worldCastDirection[1];
    gazePosition[2] += worldCastDirection[2];

    char result[] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    if (checkCollisionGPU(result, camValues[0], camValues[1], camValues[2], gazePosition[0], gazePosition[1], gazePosition[2]))
    {
        int triIdx;
        float colX, colY, colZ;

        memcpy(&triIdx, &result[ 0], 4);
        memcpy(&colX,   &result[ 4], 4);
        memcpy(&colY,   &result[ 8], 4);
        memcpy(&colZ,   &result[12], 4);

        return sampleTriangleColorGPU(colX, colY, colZ, triIdx, inTexture);
    }
    else //no collision
    {
        return pixelFromRgbGPU(0, 0, 0);
    }
    //if (res == 0)
    {
        //return pixelFromRgbGPU(0, 0, 0);
    }

    //return pixelFromRgbGPU(1, 1, 1);
    //double x = ((double)screenX)/displayWidth;
    //double y = ((double)screenY)/displayHeight;


    //return pixelFromRgb(camValues[0]/360.0, camValues[0]/360.0, camValues[0]/360.0);
    //return pixelFromRgb(abs(worldCastDirection[0]), abs(worldCastDirection[1]), abs(worldCastDirection[2]));
    //printf("%f\n", camValues[0]);
    //return pixelFromRgb(camValues[3]/360.0, (camValues[4]+180)/360.0, 1.0);

    //int texX = (int)(x*textureWidth);
    //int texY = (int)(y*textureHeight);
    //
    //int texIdx = texY*textureWidth + texX;
    //
    //return texIdx;
}

__global__ void renderPixelsGPU(int* outDisplay, const int* inTexture, const float* camValues)
{
    //int i = threadIdx.x;
    //c[i] = a[i] + b[i];
    //printf("%d, %d\n", blockIdx.x, threadIdx.x);
    int idx = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

    if (idx == 0)
    {
        //printf("camPos   = %f %f %f\n", camValues[0], camValues[1], camValues[2]);
        //printf("camYaw   = %f\n", camValues[3]);
        //printf("camPitch = %f\n\n", camValues[4]);
    }

    int totalPixels = displayWidth*displayHeight;
    int pixelsToWorkOn = totalPixels/(THREADS_PER_BLOCK*BLOCKS);

    //printf("%d\n", pixelsToWorkOn);

    for (int i = 0; i < pixelsToWorkOn; i++)
    {
        int pixelIdx = idx*pixelsToWorkOn + i;
        int screenX = pixelIdx % displayWidth;
        int screenY = pixelIdx / displayWidth;

        int color = doPixelGPU(camValues, screenX, screenY, inTexture);
        //if (textureIdx < 0 || textureIdx >= textureWidth*textureHeight)
        {
            //textureIdx = 0;
        }
        outDisplay[pixelIdx] = color;

        //int color = doPixel(camValues, triangles, screenX, screenY, (screenX == 800 && screenY == 450));
        //outDisplay[pixelIdx] = color;
    }
}

//class Test
//{
//public:
//    Test::Test()
//    {
//        
//    }
//};

int main(int argc, char* argv[])
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    SDL_Init( SDL_INIT_EVERYTHING );
    printf( "%s\n", IMG_GetError() );
    IMG_Init(IMG_INIT_PNG);
    printf( "%s\n", IMG_GetError() );

    window = SDL_CreateWindow( "Hello SDL World", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, displayWidth, displayHeight, SDL_WINDOW_SHOWN);
    
    // Check that the window was successfully created
    if (window == nullptr)
    {
        // In the case that the window could not be made...
        printf("Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    //calculate HFOV
	//double aspectRatio = (double)displayWidth / (double)displayHeight;
    //double heightOfFrustrum = 2*NEAR_PLANE*tan(Maths::toRadians(VFOV/2));
    //double widthOfFrustrum = aspectRatio*heightOfFrustrum;
    //HFOV = Maths::toDegrees(2*(atan2(widthOfFrustrum/2, NEAR_PLANE)));
    //printf("HFOV = %f\n", HFOV);

    camPosition.set(-30.7f, 3.0f, -1.66f);
    camPitch = 13.8f;
    camYaw = 91.8f;

    windowSurface = SDL_GetWindowSurface(window);

    textureImg = IMG_Load("res/img.png");
    if (textureImg == nullptr)
    {
        std::fprintf(stdout, "Error: Cannot load texture '%s'\n", "res/img.png");
    }

    cudaError_t cudaStatus = setupGlobalVars();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addWithCuda failed!");
        Sleep(1000);
        return 1;
    }

    int frameCount = 3;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < frameCount; i++)
    {
        cudaStatus = render();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addWithCuda failed!");
            Sleep(1000);
            return 1;
        }
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span = t2 - t1;
    double timeMillis = time_span.count();
    double timeSeconds = timeMillis/1000.0;
    printf("%f fps\n", frameCount/timeSeconds);
    Uint32 lastTimestamp = SDL_GetTicks();

    bool conti = true;
    while (conti)
    {
        //SDL_Delay(1);
        Uint32 newTimestamp = SDL_GetTicks();
        float dt = (newTimestamp - lastTimestamp)/1000.0f;
        lastTimestamp = newTimestamp;

        SDL_Event windowEvent;
        while (SDL_PollEvent(&windowEvent))
        {
            switch (windowEvent.type)
            {
                case SDL_QUIT:
                conti = false;
                break;

                case SDL_MOUSEMOTION:
                camYaw   =        360*(((float)windowEvent.motion.x)/displayWidth);
                camPitch = -(90 - 180*(((float)windowEvent.motion.y)/displayHeight));
                break;

                default:
                break;
            }
        }

        const Uint8* keyboardState = SDL_GetKeyboardState(nullptr);

        float c = cos(Maths::toRadians(camYaw));
        float s = sin(Maths::toRadians(camYaw));
        float speed = 20.0;

		if (keyboardState[SDL_SCANCODE_W])
		{
            float x = 0;
            float y = -speed*dt;
			float newX = x*c - y*s;
			float newZ = x*s + y*c;
			camPosition.x += newX;
			camPosition.z += newZ;
		}
		if (keyboardState[SDL_SCANCODE_S])
		{
			float x = 0;
            float y = speed*dt;
			float newX = x*c - y*s;
			float newZ = x*s + y*c;
			camPosition.x += newX;
			camPosition.z += newZ;
		}
		if (keyboardState[SDL_SCANCODE_A])
		{
			float x = -speed*dt;
            float y = 0;
			float newX = x*c - y*s;
			float newZ = x*s + y*c;
			camPosition.x += newX;
			camPosition.z += newZ;
		}
		if (keyboardState[SDL_SCANCODE_D])
		{
			float x = speed*dt;
            float y = 0;
			float newX = x*c - y*s;
			float newZ = x*s + y*c;
			camPosition.x += newX;
			camPosition.z += newZ;
		}
		if (keyboardState[SDL_SCANCODE_Z])
		{
			camPosition.y-= speed*dt;
		}
		if (keyboardState[SDL_SCANCODE_X])
		{
			camPosition.y+= speed*dt;
		}

        printf("fps = %f\n", 1/dt);

        render();

        //set_pixel(windowSurface, 32, 32, 0xFFFFFFFF);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        //Sleep(1000);
        return 1;
    }

    //Sleep(5000);
    return 0;
}

cudaError_t setupGlobalVars()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&displayBufferGPU, displayWidth * displayHeight * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&textureBufferGPU, textureWidth * textureHeight * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&camValuesGPU, 5 * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    CollisionModel* model = loadCollisionModel("res/", "sky.obj");
    std::vector<char> buf = model->serializeToGPU();
    //trianglesGPU = new char[(int)buf.size()];
    //cudaStatus = cudaMalloc((void**)&trianglesGPU, (int)buf.size() * sizeof(char));
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMalloc failed!");
    //    return cudaStatus;
    //}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(textureBufferGPU, textureImg->pixels, (textureImg->w*textureImg->h) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    //cudaStatus = cudaMemcpy(trianglesGPU, &buf[0], (int)buf.size() * sizeof(char), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    return cudaStatus;
    //}
    cudaStatus = cudaMemcpyToSymbol(trianglesGPU, &buf[0], (int)buf.size());
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyToSymbol failed!");
        return cudaStatus;
    }

    return cudaStatus;
}

cudaError_t render()
{
    cudaError_t cudaStatus;

    float camValues[5] = {camPosition.x, camPosition.y, camPosition.z, camYaw, camPitch};
    cudaMemcpy(camValuesGPU, camValues, 5 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    renderPixelsGPU CUDA_KERNEL(BLOCKS, THREADS_PER_BLOCK)(displayBufferGPU, textureBufferGPU, camValuesGPU);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    //SDL_Delay(6);
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    //cudaStatus = cudaDeviceSynchronize();
    //if (cudaStatus != cudaSuccess)
    {
        //fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        //return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(windowSurface->pixels, displayBufferGPU, displayWidth * displayHeight * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    SDL_UpdateWindowSurface(window);

//Error:
//    cudaFree(displayBufferGPU);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    Sleep(1000);
    return cudaStatus;
}

//direction of axis,
//point to rotate, 
//angle of rotation, in radians
__device__  void rotatePointGPU(float result[],
    const float point[],
	const float rotationAxis[],
	const float theta)
{
    float u = rotationAxis[0];
    float v = rotationAxis[1];
    float w = rotationAxis[2];

    float x = point[0];
    float y = point[1];
    float z = point[2];

    if (sqrt(u*u + v*v + w*w) < 0.000000001)
	{
		//printf("Warning: trying to rotate by a very small axis [%f %f %f]\n", u, v, w);
		result[0] = x;
		result[1] = y;
		result[2] = z;
		return;
	}

    float a = 0;
    float b = 0;
    float c = 0;

	float u2 = u*u;
	float v2 = v*v;
	float w2 = w*w;
	float l2 = u2 + v2 + w2;
	float l = sqrt(l2);

	float cosT = cos(theta);
	float oneMinusCosT = 1 - cosT;
	float sinT = sin(theta);

	result[0] = ((a*(v2 + w2) - u*(b*v + c*w - u*x - v*y - w*z)) * oneMinusCosT
		+ l2*x*cosT
		+ l*(-c*v + b*w - w*y + v*z)*sinT) / l2;

	result[1] = ((b*(u2 + w2) - v*(a*u + c*w - u*x - v*y - w*z)) * oneMinusCosT
		+ l2*y*cosT
		+ l*(c*u - a*w + w*x - u*z)*sinT) / l2;

	result[2] = ((c*(u2 + v2) - w*(a*u + b*v - u*x - v*y - w*z)) * oneMinusCosT
		+ l2*z*cosT
		+ l*(-b*u + a*v - v*x + u*y)*sinT) / l2;
}

//out is vector 3 of float
__device__ void calcWorldSpaceDirectionVectorFromScreenSpaceCoordsGPU(float out[], const float clickPosX, const float clickPosY, const float camYaw, const float camPitch)
{
    float aspectRatio = ((float)displayWidth)/displayHeight;

    float normalizedX = (clickPosX-(displayWidth/2))/((displayWidth/2));
    float normalizedY = (clickPosY-(displayHeight/2))/((displayHeight/2));

    float frustrumLengthY = nearPlane*tan(toRadiansGPU(fovV/2));
    float frustrumLengthX = aspectRatio*frustrumLengthY;

    float cameraSpaceCoordX = normalizedX*frustrumLengthX;
    float cameraSpaceCoordY = -normalizedY*frustrumLengthY;
    float cameraSpaceCoordZ = -nearPlane;

    float cameraSpaceDirection[] = {cameraSpaceCoordX, cameraSpaceCoordY, cameraSpaceCoordZ};
    float len = sqrt(cameraSpaceCoordX*cameraSpaceCoordX + cameraSpaceCoordY*cameraSpaceCoordY + cameraSpaceCoordZ*cameraSpaceCoordZ);

    cameraSpaceDirection[0] = cameraSpaceDirection[0]/len;
    cameraSpaceDirection[1] = cameraSpaceDirection[1]/len;
    cameraSpaceDirection[2] = cameraSpaceDirection[2]/len;

    float xAxis[] = {-1, 0, 0};
    float yAxis[] = {0, -1, 0};

    float worldSpaceOffset[] = {0, 0, 0};
    rotatePointGPU(worldSpaceOffset, cameraSpaceDirection, xAxis, toRadiansGPU(camPitch));
    rotatePointGPU(out, worldSpaceOffset, yAxis, toRadiansGPU(camYaw));
}

__device__ float inline toRadiansGPU(const float deg)
{
    return deg*0.01745329251f;
}

__device__ int inline pixelFromRgbGPU(const float r1, const float g1, const float b1)
{
    Uint32 r = (int)(255*r1);
    Uint32 g = (int)(255*g1);
    Uint32 b = (int)(255*b1);
    r = r << 16;
    g = g << 8;

    return 0xFF000000 | r | g | b;
}

__device__ bool checkPointInTriangle3DGPU(
	const float checkx, const float checky, const float checkz,
	const int triIdx)
{
    int base = triIdx*93;

    float nX, nY, nZ;
    memcpy(&nX, &trianglesGPU[56+base], 4);
    memcpy(&nY, &trianglesGPU[60+base], 4);
    memcpy(&nZ, &trianglesGPU[64+base], 4);

    nX = fabs(nX);
    nY = fabs(nY);
    nZ = fabs(nZ);

	if (nY > nX && nY > nZ)
	{
		//from the top
        float p1x, p1z, p2x, p2z, p3x, p3z;
        memcpy(&p1x, &trianglesGPU[ 4+base], 4);
        memcpy(&p1z, &trianglesGPU[12+base], 4);
        memcpy(&p2x, &trianglesGPU[16+base], 4);
        memcpy(&p2z, &trianglesGPU[24+base], 4);
        memcpy(&p3x, &trianglesGPU[28+base], 4);
        memcpy(&p3z, &trianglesGPU[36+base], 4);
		return (checkPointInTriangle2DGPU(
                checkx, checkz, 
                p1x, p1z, 
                p2x, p2z, 
                p3x, p3z));
	}
	else if (nX > nZ)
	{
		//from the left
        float p1y, p1z, p2y, p2z, p3y, p3z;
        memcpy(&p1y, &trianglesGPU[ 8+base], 4);
        memcpy(&p1z, &trianglesGPU[12+base], 4);
        memcpy(&p2y, &trianglesGPU[20+base], 4);
        memcpy(&p2z, &trianglesGPU[24+base], 4);
        memcpy(&p3y, &trianglesGPU[32+base], 4);
        memcpy(&p3z, &trianglesGPU[36+base], 4);
		return (checkPointInTriangle2DGPU(
                checkz, checky, 
                p1z, p1y, 
                p2z, p2y, 
                p3z, p3y));
	}
    else
    {
        float p1x, p1y, p2x, p2y, p3x, p3y;
        memcpy(&p1x, &trianglesGPU[ 4+base], 4);
        memcpy(&p1y, &trianglesGPU[ 8+base], 4);
        memcpy(&p2x, &trianglesGPU[16+base], 4);
        memcpy(&p2y, &trianglesGPU[20+base], 4);
        memcpy(&p3x, &trianglesGPU[28+base], 4);
        memcpy(&p3y, &trianglesGPU[32+base], 4);
        //from the front
	    return (checkPointInTriangle2DGPU(
                checkx, checky, 
                p1x, p1y, 
                p2x, p2y, 
                p3x, p3y));
    }
}

__device__ bool checkPointInTriangle2DGPU(
	const float x,  const float y,
	const float x1, const float y1,
	const float x2, const float y2,
	const float x3, const float y3)
{
	float denominator = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
	float a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator;
	float b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator;
	float c = 1 - a - b;

	return (0 <= a && a <= 1 && 0 <= b && b <= 1 && 0 <= c && c <= 1);
}
