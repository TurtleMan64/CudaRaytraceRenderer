#ifndef COLLISIONMODEL_H
#define COLLISIONMODEL_H

class Triangle3D;

#include <list>
#include <vector>

class CollisionModel
{
public:
	std::list<Triangle3D*> triangles;

	CollisionModel();

    std::vector<char> serializeToGPU();
};

#endif
