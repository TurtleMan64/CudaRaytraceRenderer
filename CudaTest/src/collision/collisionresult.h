#ifndef COLLISIONRESULT_H
#define COLLISIONRESULT_H

class Triangle3D;

#include "../toolbox/vector.h"

class CollisionResult
{
public:
    bool collided;
    Vector3d collidePosition;
	Triangle3D* collideTriangle;

	CollisionResult(bool collided, Vector3d* collidePosition, Triangle3D* collideTriangle);
};

#endif
