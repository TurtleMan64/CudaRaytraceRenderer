#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>


class Vector3d
{
public:
	float x;
	float y;
	float z;

	Vector3d();
	Vector3d(float x, float y, float z);
	Vector3d(Vector3d* other);

	float getX();
	float getY();
	float getZ();

	void set(float x, float y, float z);

	void set(Vector3d* other);

	float length();

	float lengthSquared();

	void setLength(float newLength);

	void normalize();

	void neg();

	float dot(Vector3d* other);

    float dot(Vector3d other);

	void scale(float scale);

	Vector3d scaleCopy(float scale);

	Vector3d cross(Vector3d* other);

    Vector3d cross(Vector3d other);

	Vector3d operator + (const Vector3d &other);

	Vector3d operator - (const Vector3d &other);

	Vector3d operator * (const Vector3d &other);

	Vector3d operator / (const Vector3d &other);
};

class Vector2d
{
public:
	float x;
	float y;

	Vector2d();
	Vector2d(float x, float y);
	Vector2d(Vector2d* other);

	float getX();
	float getY();

	void set(float x, float y);

	void set(Vector2d* other);

	float length();

	Vector2d normalized();

	void neg();

	float dot(Vector2d* other);

	Vector2d operator + (const Vector2d &other);

	Vector2d operator - (const Vector2d &other);

	Vector2d operator * (const Vector2d &other);

	Vector2d operator / (const Vector2d &other);

	Vector2d operator * (const float &scale);
};

class Vector4d
{
public:
	float x;
	float y;
	float z;
	float w;

	Vector4d();
	Vector4d(float x, float y, float z, float w);
	Vector4d(Vector4d* other);

	float getX();
	float getY();
	float getZ();
	float getW();

	void set(float x, float y, float z, float w);

	void set(Vector4d* other);

	float length();

	float lengthSquared();

	void normalize();

	void neg();

	float dot(Vector4d* other);

	void scale(float scale);

	Vector4d operator + (const Vector4d &other);

	Vector4d operator - (const Vector4d &other);

	Vector4d operator * (const Vector4d &other);

	Vector4d operator / (const Vector4d &other);
};
#endif
