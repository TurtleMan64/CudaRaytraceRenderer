#include <math.h>
#include <cstdio>

#include "vector.h"

float Vector3d::getX() { return x; }
float Vector3d::getY() { return y; }
float Vector3d::getZ() { return z; }

void Vector3d::set(float newX, float newY, float newZ)
{
	x = newX;
	y = newY;
	z = newZ;
}

void Vector3d::set(Vector3d* other)
{
	x = other->x;
	y = other->y;
	z = other->z;
}

float Vector3d::length()
{
	return sqrtf(x*x + y*y + z*z);
}

float Vector3d::lengthSquared()
{
	return (x * x) + (y * y) + (z * z);
}

void Vector3d::setLength(float newLength)
{
	float currLength = length();
	if (currLength > 0.0000001f)
	{
		float ratio = newLength/currLength;
		x *= ratio;
		y *= ratio;
		z *= ratio;
	}
	else 
	{
		std::fprintf(stdout, "Warning: Trying to set length of a very small vector [%f %f %f]\n", x, y, z);
		float xa = fabsf(x);
		float ya = fabsf(y);
		float max = fmaxf(xa, fmaxf(ya, fabsf(z)));
		if (xa == max)
		{
			y = 0;
			z = 0;
			if (x > 0)
			{
				x = newLength;
			}
			else
			{
				x = -newLength;
			}
		}
		else if (ya == max)
		{
			x = 0;
			z = 0;
			if (y > 0)
			{
				y = newLength;
			}
			else
			{
				y = -newLength;
			}
		}
		else
		{
			x = 0;
			y = 0;
			if (z > 0)
			{
				z = newLength;
			}
			else
			{
				z = -newLength;
			}
		}
	}
}

void Vector3d::normalize()
{
	float mag = length();

	if (mag > 0.0000001f)
	{
		x = x / mag;
		y = y / mag;
		z = z / mag;
	}
	else
	{
		std::fprintf(stdout, "Warning: Trying to normalize a very small vector [%f %f %f]\n", x, y, z);
		float xa = fabsf(x);
		float ya = fabsf(y);
		float max = fmaxf(xa, fmaxf(ya, fabsf(z)));
		if (xa == max)
		{
			y = 0;
			z = 0;
			if (x > 0)
			{
				x = 1;
			}
			else
			{
				x = -1;
			}
		}
		else if (ya == max)
		{
			x = 0;
			z = 0;
			if (y > 0)
			{
				y = 1;
			}
			else
			{
				y = -1;
			}
		}
		else
		{
			x = 0;
			y = 0;
			if (z > 0)
			{
				z = 1;
			}
			else
			{
				z = -1;
			}
		}
	}
}

void Vector3d::neg()
{
	x = -x;
	y = -y;
	z = -z;
}

void Vector3d::scale(float scale)
{
	x *= scale;
	y *= scale;
	z *= scale;
}

Vector3d Vector3d::scaleCopy(float scale)
{
	return Vector3d(x*scale, y*scale, z*scale);
}

float Vector3d::dot(Vector3d* other)
{
	return x * other->getX() + y * other->getY() + z * other->getZ();
}

float Vector3d::dot(Vector3d other)
{
	return x * other.getX() + y * other.getY() + z * other.getZ();
}

Vector3d Vector3d::cross(Vector3d* other)
{
	float x_ = y * other->getZ() - z * other->getY();
	float y_ = z * other->getX() - x * other->getZ();
	float z_ = x * other->getY() - y * other->getX();

	return Vector3d(x_, y_, z_);
}

Vector3d Vector3d::cross(Vector3d other)
{
	float x_ = y * other.getZ() - z * other.getY();
	float y_ = z * other.getX() - x * other.getZ();
	float z_ = x * other.getY() - y * other.getX();

	return Vector3d(x_, y_, z_);
}

Vector3d Vector3d::operator + (const Vector3d &other)
{
	return Vector3d(x + other.x, y + other.y, z + other.z);
}

Vector3d Vector3d::operator - (const Vector3d &other)
{
	return Vector3d(x - other.x, y - other.y, z - other.z);
}

Vector3d Vector3d::operator * (const Vector3d &other)
{
	return Vector3d(x * other.x, y * other.y, z * other.z);
}

Vector3d Vector3d::operator / (const Vector3d &other)
{
	return Vector3d(x / other.x, y / other.y, z / other.z);
}


Vector3d::Vector3d()
{
	x = 0;
	y = 0;
	z = 0;
}

Vector3d::Vector3d(float x, float y, float z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

Vector3d::Vector3d(Vector3d* base)
{
	this->x = base->x;
	this->y = base->y;
	this->z = base->z;
}






float Vector2d::getX() { return x; }
float Vector2d::getY() { return y; }

void Vector2d::set(float newX, float newY)
{
	x = newX;
	y = newY;
}

void Vector2d::set(Vector2d* vec)
{
	x = vec->x;
	y = vec->y;
}

float Vector2d::length()
{
	return sqrtf((x * x) + (y * y));
}

Vector2d Vector2d::normalized()
{
	float mag = length();

	return Vector2d(x / mag, y / mag);
}

void Vector2d::neg()
{
	x = -x;
	y = -y;
}

float Vector2d::dot(Vector2d* other)
{
	return x * other->getX() + y * other->getY();
}

Vector2d Vector2d::operator + (const Vector2d &other)
{
	return Vector2d(x + other.x, y + other.y);
}

Vector2d Vector2d::operator - (const Vector2d &other)
{
	return Vector2d(x - other.x, y - other.y);
}

Vector2d Vector2d::operator * (const Vector2d &other)
{
	return Vector2d(x * other.x, y * other.y);
}

Vector2d Vector2d::operator / (const Vector2d &other)
{
	return Vector2d(x / other.x, y / other.y);
}

Vector2d Vector2d::operator * (const float &scale)
{
	return Vector2d(x * scale, y * scale);
}

Vector2d::Vector2d()
{
	x = 0;
	y = 0;
}

Vector2d::Vector2d(float x, float y)
{
	this->x = x;
	this->y = y;
}

Vector2d::Vector2d(Vector2d* base)
{
	this->x = base->x;
	this->y = base->y;
}















float Vector4d::getX() { return x; }
float Vector4d::getY() { return y; }
float Vector4d::getZ() { return z; }
float Vector4d::getW() { return w; }

void Vector4d::set(float newX, float newY, float newZ, float newW)
{
	x = newX;
	y = newY;
	z = newZ;
	w = newW;
}

void Vector4d::set(Vector4d* other)
{
	x = other->x;
	y = other->y;
	z = other->z;
	w = other->w;
}

float Vector4d::length()
{
	return sqrtf((x * x) + (y * y) + (z * z) + (w * w));
}

float Vector4d::lengthSquared()
{
	return (x * x) + (y * y) + (z * z) + (w * w);
}

void Vector4d::normalize()
{
	float mag = length();

	x = x / mag;
	y = y / mag;
	z = z / mag;
	w = w / mag;
}

void Vector4d::neg()
{
	x = -x;
	y = -y;
	z = -z;
	w = -w;
}

void Vector4d::scale(float scale)
{
	x *= scale;
	y *= scale;
	z *= scale;
	w *= scale;
}

float Vector4d::dot(Vector4d* other)
{
	return x * other->getX() + y * other->getY() + z * other->getZ() + w * other->getW();
}

Vector4d Vector4d::operator + (const Vector4d &other)
{
	return Vector4d(x + other.x, y + other.y, z + other.z, w + other.w);
}

Vector4d Vector4d::operator - (const Vector4d &other)
{
	return Vector4d(x - other.x, y - other.y, z - other.z, w - other.w);
}

Vector4d Vector4d::operator * (const Vector4d &other)
{
	return Vector4d(x * other.x, y * other.y, z * other.z, w * other.w);
}

Vector4d Vector4d::operator / (const Vector4d &other)
{
	return Vector4d(x / other.x, y / other.y, z / other.z, w / other.w);
}


Vector4d::Vector4d()
{
	x = 0;
	y = 0;
	z = 0;
	w = 0;
}

Vector4d::Vector4d(float x, float y, float z, float w)
{
	this->x = x;
	this->y = y;
	this->z = z;
	this->w = w;
}

Vector4d::Vector4d(Vector4d* base)
{
	this->x = base->x;
	this->y = base->y;
	this->z = base->z;
	this->w = base->w;
}
