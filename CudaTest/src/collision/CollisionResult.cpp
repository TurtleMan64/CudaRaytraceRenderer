#include "collisionresult.h"
#include "triangle3d.h"
#include "../toolbox/vector.h"

CollisionResult::CollisionResult(bool collided, Vector3d* collidePosition, Triangle3D* collideTriangle)
{
    this->collided = collided;
	this->collidePosition.set(collidePosition);
    this->collideTriangle = collideTriangle;
}
