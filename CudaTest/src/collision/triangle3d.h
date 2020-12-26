#ifndef TRIANGLE3D_H
#define TRIANGLE3D_H

class Material;

#include "../toolbox/vector.h"

class Triangle3D
{
public:
	Vector3d p1;
	Vector3d p2;
	Vector3d p3;

	Vector3d normal;
	Vector3d normalSmall;

    Vector2d uv1;
    Vector2d uv2;
    Vector2d uv3;
	
	Material* material;

	float A;
	float B;
	float C;
	float D;

	float maxX;
	float minX;
	float maxY;
	float minY;
	float maxZ;
	float minZ;

	Triangle3D(Vector3d* newP1, Vector3d* newP2, Vector3d* newP3);

	void generateValues();
};

#endif
