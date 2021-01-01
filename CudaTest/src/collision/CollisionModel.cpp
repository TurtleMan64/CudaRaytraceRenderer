#include "collisionmodel.h"
#include "triangle3d.h"
#include "material.h"
#include <vector>
#include <cstring>

CollisionModel::CollisionModel()
{
	
}

//format = first 4 bytes = number of triangles
// x1, y1, z1,   //12 bytes
// x2, y2, z2,   //12 bytes
// x3, y3, z3,   //12 bytes
// A B C D       //16 bytes
// nx, ny, nz    //12 bytes
// textureId     //4  bytes
// p1u, p1v,     //8 bytes
// p2u, p2v,     //8 bytes
// p3u, p3v,     //8 bytes
// flags         //4  byte
std::vector<char> CollisionModel::serializeTrianglesToGPU()
{
    int num = (int)triangles.size();
    int totalbytes = 4 + num*96;
    char* arr = new char[totalbytes];

    memcpy(arr, &num, 4);
    int triIdx = 0;
    for (Triangle3D* tri : triangles)
    {
        int base = triIdx*96;
        memcpy(&arr[ 4+base], &tri->p1.x, 4); //p1, p2, p3
        memcpy(&arr[ 8+base], &tri->p1.y, 4);
        memcpy(&arr[12+base], &tri->p1.z, 4);
        memcpy(&arr[16+base], &tri->p2.x, 4);
        memcpy(&arr[20+base], &tri->p2.y, 4);
        memcpy(&arr[24+base], &tri->p2.z, 4);
        memcpy(&arr[28+base], &tri->p3.x, 4);
        memcpy(&arr[32+base], &tri->p3.y, 4);
        memcpy(&arr[36+base], &tri->p3.z, 4);

        memcpy(&arr[40+base], &tri->A, 4); //abcd
        memcpy(&arr[44+base], &tri->B, 4);
        memcpy(&arr[48+base], &tri->C, 4);
        memcpy(&arr[52+base], &tri->D, 4);

        memcpy(&arr[56+base], &tri->normal.x, 4); //normal
        memcpy(&arr[60+base], &tri->normal.y, 4);
        memcpy(&arr[64+base], &tri->normal.z, 4);

        int textureId = 0;
        auto it = find(materials.begin(), materials.end(), tri->material);
        if (it != materials.end())
        {
            textureId = (int)(it - materials.begin());
            //printf("textureid = %d\n", textureId);
        }
        else
        {
            printf("error: texture id not found\n");
        }
        memcpy(&arr[68+base], &textureId, 4); //texture id

        memcpy(&arr[72+base], &tri->uv1.x, 4); //uv
        memcpy(&arr[76+base], &tri->uv1.y, 4);
        memcpy(&arr[80+base], &tri->uv2.x, 4);
        memcpy(&arr[84+base], &tri->uv2.y, 4);
        memcpy(&arr[88+base], &tri->uv3.x, 4);
        memcpy(&arr[92+base], &tri->uv3.y, 4);

        memcpy(&arr[96+base], &tri->material->type, 4); //flags

        triIdx++;
    }

    std::vector<char> dat;
    for (int i = 0; i < totalbytes; i++)
    {
        dat.push_back(arr[i]);
    }
    delete[] arr;
    return dat;
}

void CollisionModel::serializeTextureToGPU(int textureId, int mapType, int* out)
{
    Material* m = materials[textureId];
    SDL_Surface* s = m->textureDiffuse;
    if (mapType == 1)
    {
        s = m->textureNormal;
    }
    for (int i = 0; i < s->w*s->h; i++)
    {
        int color;
        char* pix = (char*)s->pixels;
        memcpy(&color, &pix[i*4], 4);
        out[i] = color;
    }
}
