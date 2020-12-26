#ifndef MATHS_H
#define MATHS_H

class Camera;
class Triangle3D;

#include <random>
#include <chrono>
#include "vector.h"

class Maths
{
private:
	static std::mt19937* generatorUniform;
	static std::uniform_real_distribution<float>* distributionUniform;

	static std::default_random_engine* generatorNormal;
	static std::normal_distribution<float>* distributionNormal;

public:
	static const float PI;

	static float toRadians(float deg);

	static float toDegrees(float rad);

    //assumes the size of int and double are 32 bits
	static int sign(float value);

	//
	static float approach(float initialValue, float terminalValue, float approachConstant, float timePassed);

	//result needs to be array of 3 doubles
	//theta is in radians
	static void rotatePoint(float result[],
		float a, float b, float c,
		float u, float v, float w,
		float x, float y, float z,
		float theta);

	//Point that axis goes through,
	//direction of axis,
	//point to rotate, 
	//angle of rotation, in radians
	static Vector3d rotatePoint(
		Vector3d* pointToRotate,
		Vector3d* axisOfRotation,
		float theta);

	//Given two vectors, linear rotate from the A to B by percent and return that new vector.
	//If the two vectors are too small or are too similar already, a copy of A is retured.
	static Vector3d interpolateVector(Vector3d* A, Vector3d* B, float percent);

	//calculates the angle in radians between two vectors
	static float angleBetweenVectors(Vector3d* A, Vector3d* B);

	//given two points A and B, returns which one is closer to a point Test
	static Vector3d getCloserPoint(Vector3d* A, Vector3d* B, Vector3d* testPoint);

	static Vector3d bounceVector(Vector3d* initialVelocity, Vector3d* surfaceNormal, float elasticity);

	/** Returns the point on a sphere that has the given angles from the center
	* @param angH in radians
	* @param angV in radians
	* @param radius
	* @return
	*/
	static Vector3d spherePositionFromAngles(float angH, float angV, float radius);

    //gives the yaw and pitch (in radians) of a direction vector
    static Vector2d anglesFromDirection(Vector3d* direction);

	//Generates a uniformly distributed random position on a sphere of radius 1
	static Vector3d randomPointOnSphere();

	static Vector3d projectOntoPlane(Vector3d* A, Vector3d* normal);

	//projects a vector along a line
	static Vector3d projectAlongLine(Vector3d* A, Vector3d* line);

	//calculates an arbitrary vector that is perpendicular to the given vector vec
	static Vector3d calculatePerpendicular(Vector3d* vec);

	//returns uniform random double >= 0 and < 1
	static float random();

	//normal distribution mean = 0, std dev = 1
	static float nextGaussian();

	//returns uniform random double >= 0 and < 1
	static float nextUniform();

	//Calculate the A B C and D values for a plane from a normal and a point
	static Vector4d calcPlaneValues(Vector3d* point, Vector3d* normal);

	//Calculate the A B C and D values for a plane from 3 points
	static Vector4d calcPlaneValues(Vector3d* p1, Vector3d* p2, Vector3d* p3);

    //When you click on the screen, you are given the x and y position of the click,
    // in screen space. This function will take those coordinates and convert them into
    // a 3D vector in world space that represents a line that goes directly from the
    // camera's eye to the place you clicked. This can then be used to do things
    // like place an object where the click happend, or do a collision check.
    static Vector3d calcWorldSpaceDirectionVectorFromScreenSpaceCoords(float clickPosX, float clickPosY, int screenW, int screenH, float nearPlane, float vFOV, float camYaw, float camPitch);

    //gets you the UV coordinates of a point inside the triangle
    static Vector2d interpolateUVTriangle(Triangle3D* tri, Vector3d* f);
};

#endif
