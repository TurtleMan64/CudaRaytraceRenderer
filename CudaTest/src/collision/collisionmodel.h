#ifndef COLLISIONMODEL_H
#define COLLISIONMODEL_H

class Triangle3D;
class Material;

#include <list>
#include <vector>

class CollisionModel
{
public:
	std::list<Triangle3D*> triangles;
    std::vector<Material*> materials;

	CollisionModel();

    std::vector<char> serializeTrianglesToGPU();
    void serializeTextureToGPU(int textureId, int mapType, int* out);
};

#endif
