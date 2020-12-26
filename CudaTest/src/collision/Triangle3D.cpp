#include <math.h>
#include "triangle3d.h"
#include "../toolbox/vector.h"
#include "material.h"

Triangle3D::Triangle3D(Vector3d* newP1, Vector3d* newP2, Vector3d* newP3)
{
	p1.set(newP1);
	p2.set(newP2);
	p3.set(newP3);
	
	material = nullptr;

	generateValues();
}

void Triangle3D::generateValues()
{
	Vector3d vec1(p1.x - p3.x, p1.y - p3.y, p1.z - p3.z);
	Vector3d vec2(p2.x - p3.x, p2.y - p3.y, p2.z - p3.z);

	Vector3d cross = vec1.cross(&vec2);

	float newD = cross.x*p3.x + cross.y*p3.y + cross.z*p3.z;

	A = cross.x;
	B = cross.y;
	C = cross.z;
	D = -newD;

	float mag = sqrt(A*A + B*B + C*C);

	if (mag != 0)
	{
		normal.x = A / mag;
		normal.y = B / mag;
		normal.z = C / mag;
	}
	else
	{
		normal.x = 0;
		normal.y = 1;
		normal.z = 0;
	}
	
	normalSmall.set(&normal);
	normalSmall.scale(0.000001f);

	maxX = fmax(p1.x, fmax(p2.x, p3.x));
	minX = fmin(p1.x, fmin(p2.x, p3.x));
	maxY = fmax(p1.y, fmax(p2.y, p3.y));
	minY = fmin(p1.y, fmin(p2.y, p3.y));
	maxZ = fmax(p1.z, fmax(p2.z, p3.z));
	minZ = fmin(p1.z, fmin(p2.z, p3.z));
}
