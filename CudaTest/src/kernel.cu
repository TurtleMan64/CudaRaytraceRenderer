#include <SDL2/SDL.h> 
#include <SDL2/SDL_image.h>

#include "texture_types.h"
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
#include "collision/material.h"

#define BLOCKS 60
#define THREADS_PER_BLOCK (640)
//1920 9600/15
SDL_Window* window = nullptr;

SDL_Surface* windowSurface = nullptr;
int* displayBufferGPU = nullptr;
#define displayWidth 1600
#define displayHeight 900

//temporary hard coded texture
//SDL_Surface* textureImg = nullptr;
//int* textureBufferGPU = nullptr;
//#define textureWidth 512
//#define textureHeight 512

//Texture reference for 2D int texture
#define NUM_TEXTURES 4
texture<int, 2, cudaReadModeElementType> texture1DiffuseGPU;
texture<int, 2, cudaReadModeElementType> texture1NormalGPU;
texture<int, 2, cudaReadModeElementType> texture2DiffuseGPU;
texture<int, 2, cudaReadModeElementType> texture2NormalGPU;
texture<int, 2, cudaReadModeElementType> texture3DiffuseGPU;
texture<int, 2, cudaReadModeElementType> texture3NormalGPU;
texture<int, 2, cudaReadModeElementType> texture4DiffuseGPU;
texture<int, 2, cudaReadModeElementType> texture4NormalGPU;

//cam stuff
Vector3d camPosition;
float camYaw = 91.0f;  //in degrees
float camPitch = -1.0f; //in degrees

float* camValuesGPU = nullptr; //position, yaw, pitch
#define fovV 60.0f
#define fovH 91.492844f
#define nearPlane 0.15f

//list of triangles data struct
__constant__ char trianglesGPU[32000]; //can do ~300 triangles

cudaError_t setupGlobalVars();

cudaError_t render();

__device__ inline float vec3dot(const float a[], const float b[])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ inline void vec3bounce(float A[], float normal[])
{
    float scale = -2.0f*vec3dot(A, normal);
    A[0] += normal[0]*scale;
    A[1] += normal[1]*scale;
    A[2] += normal[2]*scale;
}

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

    out[0] = -(uv1x*a1 + uv2x*a2 + uv3x*a3);
    out[1] = uv1y*a1 + uv2y*a2 + uv3y*a3;
}

__device__ void barryUvToUv(const int triIdx, const float bu, const float bv, float* outU, float* outV)
{
    int base = triIdx*96;

    float uv1x, uv1y, uv2x, uv2y, uv3x, uv3y;
    memcpy(&uv1x, &trianglesGPU[72+base], 4); //uv
    memcpy(&uv1y, &trianglesGPU[76+base], 4);
    memcpy(&uv2x, &trianglesGPU[80+base], 4);
    memcpy(&uv2y, &trianglesGPU[84+base], 4);
    memcpy(&uv3x, &trianglesGPU[88+base], 4);
    memcpy(&uv3y, &trianglesGPU[92+base], 4);

    float bw = 1-bu-bv;

    float a1 = -bw;
    float a2 = -bu;
    float a3 = -bv;

    *outU = -(uv1x*a1 + uv2x*a2 + uv3x*a3);
    *outV =   uv1y*a1 + uv2y*a2 + uv3y*a3;
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

__device__ int sampleTexture(const float u, const float v, const int textureId, const int type)
{
    switch (type)
    {
    case 0:
        switch (textureId)
        {
            case  0: return tex2D(texture1DiffuseGPU, u, v);
            case  1: return tex2D(texture2DiffuseGPU, u, v);
            case  2: return tex2D(texture3DiffuseGPU, u, v);
            case  3: return tex2D(texture4DiffuseGPU, u, v);
            default: return tex2D(texture1DiffuseGPU, u, v);
        }

    default:
        switch (textureId)
        {
            case  0: return tex2D(texture1NormalGPU, u, v);
            case  1: return tex2D(texture2NormalGPU, u, v);
            case  2: return tex2D(texture3NormalGPU, u, v);
            case  3: return tex2D(texture4NormalGPU, u, v);
            default: return tex2D(texture1NormalGPU, u, v);
        }
    }
}

__device__ void sampleTriangleUvGPU(const float cx, const float cy, const float cz, const int triIdx, float* u, float* v, bool debug)
{
    int base = triIdx*96;
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

    *u = uv[0];
    *v = uv[1];
    //int textureId = 0;
    //memcpy(&textureId, &trianglesGPU[68+base], 4); //texture id
    //switch (textureId)
    //{
    //    case  0: return tex2D(texture1DiffuseGPU, uv[0], uv[1]);
    //    case  1: return tex2D(texture2DiffuseGPU, uv[0], uv[1]);
    //    case  2: return tex2D(texture3DiffuseGPU, uv[0], uv[1]);
    //    default: return tex2D(texture1DiffuseGPU, uv[0], uv[1]);
    //}
    //int x = (int)(textureWidth*uv[0]);
    //int y = (int)(textureHeight*uv[1]);
    //x = x % textureWidth;
    //y = y % textureHeight;
    //while (x < 0)
    //{
    //    x = x + textureWidth;
    //}
    //
    //while (y < 0)
    //{
    //    y = y + textureHeight;
    //}
    //return inTexture[y*textureWidth + x];
}

__device__ void calculateTangentSpaceOfUv(const int triIdx, float* outTangent, float* outBitangent)
{
    int base = triIdx*96;
    float p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, uv1x, uv1y, uv2x, uv2y, uv3x, uv3y;
    memcpy(&p1x, &trianglesGPU[ 4+base], 4);
    memcpy(&p1y, &trianglesGPU[ 8+base], 4);
    memcpy(&p1z, &trianglesGPU[12+base], 4);
    memcpy(&p2x, &trianglesGPU[16+base], 4);
    memcpy(&p2y, &trianglesGPU[20+base], 4);
    memcpy(&p2z, &trianglesGPU[24+base], 4);
    memcpy(&p3x, &trianglesGPU[28+base], 4);
    memcpy(&p3y, &trianglesGPU[32+base], 4);
    memcpy(&p3z, &trianglesGPU[36+base], 4);

    memcpy(&uv1x, &trianglesGPU[72+base], 4); //uv
    memcpy(&uv1y, &trianglesGPU[76+base], 4);
    memcpy(&uv2x, &trianglesGPU[80+base], 4);
    memcpy(&uv2y, &trianglesGPU[84+base], 4);
    memcpy(&uv3x, &trianglesGPU[88+base], 4);
    memcpy(&uv3y, &trianglesGPU[92+base], 4);

    float dv1[] = {p2x-p1x, p2y-p1y, p2z-p1z};
    float dv2[] = {p3x-p1x, p3y-p1y, p3z-p1z};

    float duv1[] = {uv2x-uv1x, uv2y-uv1y};
    float duv2[] = {uv3x-uv1x, uv3y-uv1y};

    float r = 1.0f / (duv1[0]*duv2[1] - duv1[1]*duv2[0]);

    float tangent[]   = { (dv1[0]*duv2[1] - dv2[0]*duv1[1])*r, (dv1[1]*duv2[1] - dv2[1]*duv1[1])*r, (dv1[2]*duv2[1] - dv2[2]*duv1[1])*r };
    outTangent[0] = tangent[0];
    outTangent[1] = tangent[1];
    outTangent[2] = tangent[2];

    float bitangent[] = { (dv2[0]*duv1[0] - dv1[0]*duv2[0])*r, (dv2[1]*duv1[0] - dv1[1]*duv2[0])*r, (dv2[2]*duv1[0] - dv1[2]*duv2[0])*r };
    outBitangent[0] = bitangent[0];
    outBitangent[1] = bitangent[1];
    outBitangent[2] = bitangent[2];
}

__device__ bool checkCollision2GPU(char* result,
                                   const float px1, const float py1, const float pz1,
                                   const float px2, const float py2, const float pz2)
{
    //float R[] = {px1, py1, pz1};
    float Rdir[] = {px2-px1, py2-py1, pz2-pz1};
    vec3normalize(Rdir);

    int numTriangles = 0;
    memcpy(&numTriangles, trianglesGPU, 4);

    float minDist = 100000000000.0f;
    int triangleId = -1;
    float cPos[] = {0,0,0};
    float bubv[] = {0,0};

    //int i = blockIdx.x;
    int i = -1;
    for (int count = 0; count < numTriangles; count++)
    {
        //i = (i+1) % numTriangles;
        i++;

        int base = i*96;
        float t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
        memcpy(&t1x, &trianglesGPU[ 4+base], 4);
        memcpy(&t1y, &trianglesGPU[ 8+base], 4);
        memcpy(&t1z, &trianglesGPU[12+base], 4);
        memcpy(&t2x, &trianglesGPU[16+base], 4);
        memcpy(&t2y, &trianglesGPU[20+base], 4);
        memcpy(&t2z, &trianglesGPU[24+base], 4);
        memcpy(&t3x, &trianglesGPU[28+base], 4);
        memcpy(&t3y, &trianglesGPU[32+base], 4);
        memcpy(&t3z, &trianglesGPU[36+base], 4);

        //float A[] = {t1x, t1y, t1z};
        //float B[] = {t2x, t2y, t2z};
        //float C[] = {t3x, t3y, t3z};

        float E1[] = {t2x-t1x, t2y-t1y, t2z-t1z};
        float E2[] = {t3x-t1x, t3y-t1y, t3z-t1z};

        float N[] = {0,0,0};
        vec3cross(N, E1, E2);

        float det = -(vec3dot(Rdir, N));
        float invdet = 1.0f/det;

        float AO[] = {px1-t1x, py1-t1y, pz1-t1z};
        float DAO[] = {0,0,0};
        vec3cross(DAO, AO, Rdir);

        float u =  vec3dot(E2, DAO)*invdet;
        float v = -vec3dot(E1, DAO)*invdet;
        float t =  vec3dot(AO,   N)*invdet;

        bool intersects = (det >= 1e-6f && t >= 0.0f && u >= 0.0f && v >= 0.0f && (u+v) <= 1.0f);

        if (intersects && t < minDist)
        {
            cPos[0] = px1 + t*Rdir[0];
			cPos[1] = py1 + t*Rdir[1];
			cPos[2] = pz1 + t*Rdir[2];

            bubv[0] = u;
            bubv[1] = v;

			minDist = t;
            triangleId = i;
		}
    }

    if (triangleId >= 0)
    {
        memcpy(&result[ 0], &triangleId, 4);
        memcpy(&result[ 4], &cPos[0], 4);
        memcpy(&result[ 8], &cPos[1], 4);
        memcpy(&result[12], &cPos[2], 4);
        memcpy(&result[16], &bubv[0], 4);
        memcpy(&result[20], &bubv[1], 4);
        return true;
    }

    return false;
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

    //int i = blockIdx.x;
    int i = 0;
    for (int count = 0; count < numTriangles; count++)
    {
        i = (i+1) % numTriangles;
        //i++;

        int base = i*96;

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

__device__ int doPixelGPU(const float* camValues, const int screenX, const int screenY)
{
    float worldCastDirection[] = {0, 0, 0};
    calcWorldSpaceDirectionVectorFromScreenSpaceCoordsGPU(worldCastDirection, (float)screenX, (float)screenY, camValues[3], camValues[4]);
    worldCastDirection[0] *= 1000;
    worldCastDirection[1] *= 1000;
    worldCastDirection[2] *= 1000;

    float startPosition[] = {camValues[0], camValues[1], camValues[2]};
    float gazePosition[]  = {camValues[0], camValues[1], camValues[2]};
    gazePosition[0] += worldCastDirection[0];
    gazePosition[1] += worldCastDirection[1];
    gazePosition[2] += worldCastDirection[2];

    char result[] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    //if (screenX == 1600/2 && screenY == 900/2)
    {
        //printf("\n");
    }
    int tries = 6;
    while (tries > 0)
    {
        tries--;
        //if (screenX == 1600/2 && screenY == 900/2)
        {
            //printf("v %f %f %f\nv %f %f %f\n", startPosition[0], startPosition[1], startPosition[2], gazePosition[0], gazePosition[1], gazePosition[2]);
        }
        if (checkCollision2GPU(result, startPosition[0], startPosition[1], startPosition[2], gazePosition[0], gazePosition[1], gazePosition[2]))
        {
            int triIdx;
            float colX, colY, colZ, bu, bv;

            memcpy(&triIdx, &result[ 0], 4);
            memcpy(&colX,   &result[ 4], 4);
            memcpy(&colY,   &result[ 8], 4);
            memcpy(&colZ,   &result[12], 4);
            memcpy(&bu,     &result[16], 4);
            memcpy(&bv,     &result[20], 4);

            //bool debug = (screenX == 1600/2 && screenY == 900/2);

            float u, v;
            //sampleTriangleUvGPU(colX, colY, colZ, triIdx, &u, &v);
            barryUvToUv(triIdx, bu, bv, &u, &v);

            int base = triIdx*96;

            char type = trianglesGPU[base + 96];

            int textureId = 0;
            memcpy(&textureId, &trianglesGPU[base + 68], 4);

            if (type == 0) //normal type
            {
                return sampleTexture(u, v, textureId, 0);
            }
            else //mirror type
            {
                int surfNormal = sampleTexture(u, v, textureId, 1);
                //float tangentNormal[] = 
                //{
                //    (((surfNormal & 0x00FF0000) >> 16) - 128)/256.0f,
                //    (((surfNormal & 0x000000FF) >>  0) -   0)/256.0f,
                //    (((surfNormal & 0x0000FF00) >>  8) - 128)/256.0f
                //};
                float tangentNormal[] = 
                {
                    (((surfNormal & 0x00FF0000) >> 16) - 128)/256.0f,
                    (((surfNormal & 0x0000FF00) >>  8) - 128)/256.0f,
                    (((surfNormal & 0x000000FF) >>  0) - 128)/256.0f
                };
                vec3normalize(tangentNormal);

                float tangent[] = {0,0,0};
                float bitangent[] = {0,0,0};
                calculateTangentSpaceOfUv(triIdx, tangent, bitangent);
                //if (screenX == 1600/2 && screenY == 900/2)
                {
                    //printf("%f %f %f       ", tangent[0], tangent[1], tangent[2]);
                    //printf("%f %f %f\n", bitangent[0], bitangent[1], bitangent[2]);
                }

                float normalTriangle[] = {0,0,0};
                memcpy(&normalTriangle[0], &trianglesGPU[56+base], 4); //normal
                memcpy(&normalTriangle[1], &trianglesGPU[60+base], 4);
                memcpy(&normalTriangle[2], &trianglesGPU[64+base], 4);

                float newNormLen = (vec3len(tangent) + vec3len(bitangent))/2;
                normalTriangle[0]*=newNormLen;
                normalTriangle[1]*=newNormLen;
                normalTriangle[2]*=newNormLen;

                float normalWorldSpace[] = {0,0,0};
                normalWorldSpace[0] = tangent[0]*tangentNormal[0] + bitangent[0]*tangentNormal[1] + normalTriangle[0]*tangentNormal[2];
                normalWorldSpace[1] = tangent[1]*tangentNormal[0] + bitangent[1]*tangentNormal[1] + normalTriangle[1]*tangentNormal[2];
                normalWorldSpace[2] = tangent[2]*tangentNormal[0] + bitangent[2]*tangentNormal[1] + normalTriangle[2]*tangentNormal[2];

                //if (screenX == 1600/2 && screenY == 900/2)
                {
                    //printf("    normalworldspace = %f %f %f\n\n", normalWorldSpace[0], normalWorldSpace[1], normalWorldSpace[2]);
                }

                //normalWorldSpace[0] = (normalWorldSpace[0]+1)/2.0f;
                //normalWorldSpace[1] = (normalWorldSpace[1]+1)/2.0f;
                //normalWorldSpace[2] = (normalWorldSpace[2]+1)/2.0f;

                //if (screenX == 1600/2 && screenY == 900/2)
                //{
                //    printf("%f %f %f\n", tangentUv[0], tangentUv[1], tangentUv[2]);
                //}
                //
                //return pixelFromRgbGPU(normalWorldSpace[0], normalWorldSpace[1], normalWorldSpace[2]);

                float newDirection[] = {gazePosition[0] - startPosition[0], gazePosition[1] - startPosition[1], gazePosition[2] - startPosition[2]};
                vec3normalize(newDirection);
                vec3normalize(normalWorldSpace);
                vec3bounce(newDirection, normalWorldSpace);

                vec3normalize(normalTriangle);
                if (vec3dot(newDirection, normalTriangle) < 0.0f) //if we reflect "inside" the triangle, flip it to outside
                {
                    vec3bounce(newDirection, normalTriangle);
                    //return pixelFromRgbGPU(1, 0, 0);
                }
                
                startPosition[0] = colX + 0.001f*newDirection[0];
                startPosition[1] = colY + 0.001f*newDirection[1];
                startPosition[2] = colZ + 0.001f*newDirection[2];
                
                gazePosition[0] = colX + 1000*newDirection[0];
                gazePosition[1] = colY + 1000*newDirection[1];
                gazePosition[2] = colZ + 1000*newDirection[2];
            }

            //return sampleTriangleColorGPU(colX, colY, colZ, triIdx);
        }
        else //no collision
        {
            return pixelFromRgbGPU(0, 0, 0);
        }
    }
    //if (res == 0)
    {
        //return pixelFromRgbGPU(0, 0, 0);
    }

    return pixelFromRgbGPU(0, 0, 0);

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

__global__ void renderPixelsGPU(int* outDisplay, const float* camValues)
{
    //int i = threadIdx.x;
    //c[i] = a[i] + b[i];
    //printf("%d, %d\n", blockIdx.x, threadIdx.x);
    int idx = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

    //if (idx == 0)
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

        int color = doPixelGPU(camValues, screenX, screenY);
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

    //textureImg = IMG_Load("res/img.png");
    //if (textureImg == nullptr)
    {
        //std::fprintf(stdout, "Error: Cannot load texture '%s'\n", "res/img.png");
    }

    cudaError_t cudaStatus = setupGlobalVars();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "renderPixelsGPU failed!");
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
            fprintf(stderr, "renderPixelsGPU failed!");
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
    Uint32 lastSecond = SDL_GetTicks();
    int fpsCount = 0;

    bool conti = true;
    while (conti)
    {
        //SDL_Delay(1);
        Uint32 newTimestamp = SDL_GetTicks();
        float dt = (newTimestamp - lastTimestamp)/1000.0f;
        lastTimestamp = newTimestamp;
        fpsCount++;
        if (newTimestamp - lastSecond >= 1000)
        {
            printf("fps = %d\n", fpsCount);
            fpsCount = 0;
            lastSecond = newTimestamp;
        }

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
        float speed = 30.0;

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

        //printf("fps = %f\n", 1/dt);

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

int pixelFromRgb(const float r1, const float g1, const float b1)
{
    Uint32 r = (int)(255*r1);
    Uint32 g = (int)(255*g1);
    Uint32 b = (int)(255*b1);
    r = r << 16;
    g = g << 8;

    return 0xFF000000 | r | g | b;
}

cudaError_t setupTextures(CollisionModel* model)
{
    cudaError_t cudaStatus;

    for (int i = 0; i < NUM_TEXTURES; i++)
    {
        for (int map = 0; map < 2; map++)
        {
            SDL_Surface* s = model->materials[i]->textureDiffuse;
            if (map == 1)
            {
                s = model->materials[i]->textureNormal;
            }
            int m1 = s->h; // height
            int n1 = s->w; // width  = #columns
            size_t pitch1, tex_ofs1;
            int* arr1 = new int[n1*m1];
            model->serializeTextureToGPU(i, map, arr1);
            int* arr_d1 = nullptr;
            cudaStatus = cudaMallocPitch((void**)&arr_d1, &pitch1, n1*sizeof(*arr_d1), m1);
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMallocPitch failed!"); return cudaStatus; }
            cudaStatus = cudaMemcpy2D(arr_d1, pitch1, arr1, n1*sizeof(int), n1*sizeof(int), m1, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2D failed!"); return cudaStatus; }
            switch (i)
            {
            case 0:
                switch (map)
                {
                case 0:
                    texture1DiffuseGPU.addressMode[0] = cudaAddressModeWrap;
                    texture1DiffuseGPU.addressMode[1] = cudaAddressModeWrap;
                    texture1DiffuseGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture1DiffuseGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture1DiffuseGPU, arr_d1, &texture1DiffuseGPU.channelDesc, n1, m1, pitch1);
                    break;
                default:
                    texture1NormalGPU.addressMode[0] = cudaAddressModeWrap;
                    texture1NormalGPU.addressMode[1] = cudaAddressModeWrap;
                    texture1NormalGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture1NormalGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture1NormalGPU, arr_d1, &texture1NormalGPU.channelDesc, n1, m1, pitch1);
                    break;
                }
                break;

            case 1:
                switch (map)
                {
                case 0:
                    texture2DiffuseGPU.addressMode[0] = cudaAddressModeWrap;
                    texture2DiffuseGPU.addressMode[1] = cudaAddressModeWrap;
                    texture2DiffuseGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture2DiffuseGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture2DiffuseGPU, arr_d1, &texture2DiffuseGPU.channelDesc, n1, m1, pitch1);
                    break;
                default:
                    texture2NormalGPU.addressMode[0] = cudaAddressModeWrap;
                    texture2NormalGPU.addressMode[1] = cudaAddressModeWrap;
                    texture2NormalGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture2NormalGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture2NormalGPU, arr_d1, &texture2NormalGPU.channelDesc, n1, m1, pitch1);
                    break;
                }
                break;

            case 2:
                switch (map)
                {
                case 0:
                    texture3DiffuseGPU.addressMode[0] = cudaAddressModeWrap;
                    texture3DiffuseGPU.addressMode[1] = cudaAddressModeWrap;
                    texture3DiffuseGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture3DiffuseGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture3DiffuseGPU, arr_d1, &texture3DiffuseGPU.channelDesc, n1, m1, pitch1);
                    break;
                default:
                    texture3NormalGPU.addressMode[0] = cudaAddressModeWrap;
                    texture3NormalGPU.addressMode[1] = cudaAddressModeWrap;
                    texture3NormalGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture3NormalGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture3NormalGPU, arr_d1, &texture3NormalGPU.channelDesc, n1, m1, pitch1);
                    break;
                }
                break;

            case 3:
                switch (map)
                {
                case 0:
                    texture4DiffuseGPU.addressMode[0] = cudaAddressModeWrap;
                    texture4DiffuseGPU.addressMode[1] = cudaAddressModeWrap;
                    texture4DiffuseGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture4DiffuseGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture4DiffuseGPU, arr_d1, &texture4DiffuseGPU.channelDesc, n1, m1, pitch1);
                    break;
                default:
                    texture4NormalGPU.addressMode[0] = cudaAddressModeWrap;
                    texture4NormalGPU.addressMode[1] = cudaAddressModeWrap;
                    texture4NormalGPU.filterMode = cudaFilterModePoint; //can only do linear with float type
                    texture4NormalGPU.normalized = true;
                    cudaStatus = cudaBindTexture2D(&tex_ofs1, &texture4NormalGPU, arr_d1, &texture4NormalGPU.channelDesc, n1, m1, pitch1);
                    break;
                }
                break;

            default:
                break;

            }
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaBindTexture2D failed!"); return cudaStatus;}
            if (tex_ofs1 !=0) { printf ("tex_ofs = %zu\n", tex_ofs1); return cudaStatus;}
        }
    }

    return cudaStatus;
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

    //cudaStatus = cudaMalloc((void**)&textureBufferGPU, textureWidth * textureHeight * sizeof(int));
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMalloc failed!");
    //    return cudaStatus;
    //}
    //texturesGPU
    //setupTextures();

    cudaStatus = cudaMalloc((void**)&camValuesGPU, 5 * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    CollisionModel* model = loadCollisionModel("res/", "Orca.obj");
    std::vector<char> buf = model->serializeTrianglesToGPU();
    setupTextures(model);
    //trianglesGPU = new char[(int)buf.size()];
    //cudaStatus = cudaMalloc((void**)&trianglesGPU, (int)buf.size() * sizeof(char));
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMalloc failed!");
    //    return cudaStatus;
    //}

    // Copy input vectors from host memory to GPU buffers.
    //cudaStatus = cudaMemcpy(textureBufferGPU, textureImg->pixels, (textureImg->w*textureImg->h) * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    return cudaStatus;
    //}

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
    cudaStatus = cudaMemcpy(camValuesGPU, camValues, 5 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy cam values failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    
    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    renderPixelsGPU CUDA_KERNEL(BLOCKS, THREADS_PER_BLOCK)(displayBufferGPU, camValuesGPU);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "renderPixelsGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
        fprintf(stderr, "cudaMemcpy display to host failed: %s\n", cudaGetErrorString(cudaStatus));
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

    if (sqrt(u*u + v*v + w*w) < 0.000000001f)
	{
		//printf("Warning: trying to rotate by a very small axis [%f %f %f]\n", u, v, w);
		result[0] = x;
		result[1] = y;
		result[2] = z;
		return;
	}

	float u2 = u*u;
	float v2 = v*v;
	float w2 = w*w;
	float l2 = u2 + v2 + w2;
	float l = sqrt(l2);

	float cosT = cos(theta);
	float oneMinusCosT = 1 - cosT;
	float sinT = sin(theta);

	result[0] = ((-u*(-u*x - v*y - w*z)) * oneMinusCosT
        + l2*x*cosT
		+ l*(-w*y + v*z)*sinT) / l2;

	result[1] = ((-v*(-u*x - v*y - w*z)) * oneMinusCosT
		+ l2*y*cosT
		+ l*( w*x - u*z)*sinT) / l2;

	result[2] = ((-w*(-u*x - v*y - w*z)) * oneMinusCosT
		+ l2*z*cosT
		+ l*(-v*x + u*y)*sinT) / l2;
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
    int base = triIdx*96;

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
