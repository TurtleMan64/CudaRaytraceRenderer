#include <cmath>
#include <random>
#include <chrono>

#include "vector.h"
#include "maths.h"
#include "../collision/triangle3d.h"

std::mt19937* Maths::generatorUniform = new std::mt19937(0);
std::uniform_real_distribution<float>* Maths::distributionUniform = new std::uniform_real_distribution<float>(0.0f, 1.0);

std::default_random_engine* Maths::generatorNormal = new std::default_random_engine(0);
std::normal_distribution<float>* Maths::distributionNormal = new std::normal_distribution<float>(0.0f, 1.0);

const float Maths::PI = 3.14159265358979323846f;

float Maths::toRadians(float degrees)
{
	return (degrees*0.01745329251f);
}

float Maths::toDegrees(float radians)
{
	return (radians*57.2957795131f);
}

int Maths::sign(float value)
{
    //int v = *(int*)&value; //get bits of value casted as an int

    //return 1 - 2*((v>>31) & 0x01);

	if (value > 0)
	{
		return 1;
	}
	else if (value < 0)
	{
		return -1;
	}
	return 0;
}

float Maths::approach(float initialValue, float terminalValue, float approachConstant, float timePassed)
{
	return ((initialValue-terminalValue)*powf(2.718281828459f, -approachConstant*timePassed) + terminalValue);
}

//Equation from https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas
//Point that axis goes through,
//direction of axis,
//point to rotate, 
//angle of rotation
void Maths::rotatePoint(float result[],
	float a, float b, float c,
	float u, float v, float w,
	float x, float y, float z,
	float theta)
{
	if (sqrt(u*u + v*v + w*w) < 0.000000001)
	{
		std::fprintf(stdout, "Warning: trying to rotate by a very small axis [%f %f %f]\n", u, v, w);
		result[0] = x;
		result[1] = y;
		result[2] = z;
		return;
	}

	float u2 = u*u;
	float v2 = v*v;
	float w2 = w*w;
	float l2 = u2 + v2 + w2;
	float l = sqrt(l2);

	float cosT = cos(theta);
	float oneMinusCosT = 1 - cosT;
	float sinT = sin(theta);

	result[0] = ((a*(v2 + w2) - u*(b*v + c*w - u*x - v*y - w*z)) * oneMinusCosT
		+ l2*x*cosT
		+ l*(-c*v + b*w - w*y + v*z)*sinT) / l2;

	result[1] = ((b*(u2 + w2) - v*(a*u + c*w - u*x - v*y - w*z)) * oneMinusCosT
		+ l2*y*cosT
		+ l*(c*u - a*w + w*x - u*z)*sinT) / l2;

	result[2] = ((c*(u2 + v2) - w*(a*u + b*v - u*x - v*y - w*z)) * oneMinusCosT
		+ l2*z*cosT
		+ l*(-b*u + a*v - v*x + u*y)*sinT) / l2;
}

//Point that axis goes through,
//direction of axis,
//point to rotate, 
//angle of rotation, in radians
Vector3d Maths::rotatePoint(
	Vector3d* pointToRotate,
	Vector3d* axisOfRotation,
	float theta)
{
	float result[3];
	Maths::rotatePoint(result, 0, 0, 0, 
		axisOfRotation->x, 
		axisOfRotation->y, 
		axisOfRotation->z, 
		pointToRotate->x, 
		pointToRotate->y, 
		pointToRotate->z, 
		theta);

	return Vector3d(result[0], result[1], result[2]);
}

Vector3d Maths::interpolateVector(Vector3d* A, Vector3d* B, float percent)
{
	Vector3d perpen = A->cross(B);
	float dotProduct = A->dot(B);
	float mag = A->length()*B->length();

	if (mag < 0.0000001)
	{
		std::fprintf(stdout, "Warning: Trying to interpolate between small vectors\n");
		return Vector3d(A);
	}

	if (dotProduct/mag > 0.99999) //Vectors are extremely similar already, just return A
	{
        std::fprintf(stdout, "Warning: Vectors are extremely similar already, just return A\n");
		return Vector3d(A);
	}

	float angle = acos(dotProduct/mag);
	percent = fminf(1.0, fmaxf(0.0, percent));
	return Maths::rotatePoint(A, &perpen, angle*percent);
}

float Maths::angleBetweenVectors(Vector3d* A, Vector3d* B)
{
	float dotProduct = A->dot(B);
	float mag = A->length()*B->length();

	if (mag < 0.0000001)
	{
        std::fprintf(stdout, "Warning: magnitude is really small\n");
		return 0;
	}

	if (dotProduct/mag > 0.999999) //Vectors are extremely similar already, just return 0
	{
        std::fprintf(stdout, "Warning: Vectors are extremely similar already, just return 0\n");
		return 0;
	}

	return acos(dotProduct/mag);
}

Vector3d Maths::getCloserPoint(Vector3d* A, Vector3d* B, Vector3d* testPoint)
{
	Vector3d deltaA(A);
	deltaA = deltaA - testPoint;
	Vector3d deltaB(B);
	deltaB = deltaB - testPoint;
	float distA = deltaA.lengthSquared();
	float distB = deltaB.lengthSquared();
	if (distA < distB)
	{
		return Vector3d(A);
	}
	return Vector3d(B);
}

//https://stackoverflow.com/questions/11132681/what-is-a-formula-to-get-a-vector-perpendicular-to-another-vector
Vector3d Maths::calculatePerpendicular(Vector3d* vec)
{
	bool b0 = (vec->x <  vec->y) && (vec->x <  vec->z);
	bool b1 = (vec->y <= vec->x) && (vec->y <  vec->z);
	bool b2 = (vec->z <= vec->x) && (vec->z <= vec->y);
	Vector3d differentVec((float)(b0), (float)(b1), (float)(b2));
	differentVec.setLength(1);
	Vector3d perpen = vec->cross(&differentVec);
	perpen.setLength(1);
	return perpen;
}

Vector3d Maths::projectAlongLine(Vector3d* A, Vector3d* line)
{
	Vector3d master(A);

	Vector3d perpen1 = Maths::calculatePerpendicular(line);
	perpen1.normalize();

	Vector3d perpen2 = perpen1.cross(line);
	perpen2.normalize();

	master = Maths::projectOntoPlane(&master, &perpen1);

	master = Maths::projectOntoPlane(&master, &perpen2);

	return master;
}

/**
* @param initialVelocity
* @param surfaceNormal
* @param elasticity Scale of the resulting vector relative to the original velocity
*/
Vector3d Maths::bounceVector(Vector3d* initialVelocity, Vector3d* surfaceNormal, float elasticity)
{
	Vector3d twoNtimesVdotN(surfaceNormal);
	twoNtimesVdotN.scale(-2 * initialVelocity->dot(surfaceNormal));

	Vector3d Vnew = (twoNtimesVdotN + initialVelocity);
	Vnew.scale(elasticity);

	return Vnew;
}


//Projects vector A to be perpendicular to vector normal
Vector3d Maths::projectOntoPlane(Vector3d* A, Vector3d* normal)
{
	Vector3d B(0, 0, 0);
	Vector3d C(A);
	Vector3d N(normal->x, normal->y, normal->z);
	N.normalize();

	N.scale(C.dot(&N));
	B = C - N;

	return B;
}

/** Returns the point on a sphere that has the given angles from the center
* @param angH in radians
* @param angV in radians
* @param radius
* @return
*/
Vector3d Maths::spherePositionFromAngles(float angH, float angV, float radius)
{
	float y   = (radius*sin(angV));
	float hpt = (radius*cos(angV));

	float x = (hpt*cos(angH));
	float z = (hpt*sin(angH));

	return Vector3d(x, y, z);
}

Vector2d Maths::anglesFromDirection(Vector3d* dir)
{
    float yaw = atan2(dir->z, dir->x);

    float hdist = sqrt(dir->z*dir->z + dir->x*dir->x);
    float pitch = atan2(dir->y, hdist);

    return Vector2d(yaw, pitch);
}

Vector3d Maths::randomPointOnSphere()
{
	float z   = Maths::nextUniform()*2 - 1;
	float lng = Maths::nextUniform()*2*Maths::PI;

	float radius = sqrt(1-(z)*(z));

	float x = radius*cos(lng);
	float y = radius*sin(lng);

	return Vector3d(x, y, z);


	//Makes a 3D Bernoulli Lemniscate ???

	//double theta = 2 * M_PI * Maths::nextUniform();
    //double phi = acosf(1 - 2 * Maths::nextUniform());
    //double x = sinf(phi) * cosf(theta);
    //double y = sinf(phi) * sinf(theta);
    //double z = cosf(phi);

	//return spherePositionFromAngles(x, y, z);
}

float Maths::random()
{
	return (rand() % RAND_MAX) / ((float)(RAND_MAX));
}

float Maths::nextGaussian()
{
	return (*Maths::distributionNormal)(*Maths::generatorNormal);
}

float Maths::nextUniform()
{
	return (*Maths::distributionUniform)(*Maths::generatorUniform);
}

Vector4d Maths::calcPlaneValues(Vector3d* p1, Vector3d* p2, Vector3d* p3)
{
	Vector3d vec1(p1->x - p3->x, p1->y - p3->y, p1->z - p3->z);
	Vector3d vec2(p2->x - p3->x, p2->y - p3->y, p2->z - p3->z);

	Vector3d cross = vec1.cross(&vec2);

	float newD = cross.x*p3->x + cross.y*p3->y + cross.z*p3->z;

	return Vector4d(cross.x, cross.y, cross.z, -newD);
}

Vector4d Maths::calcPlaneValues(Vector3d* point, Vector3d* normal)
{
	Vector3d perp1 = Maths::calculatePerpendicular(normal);
	Vector3d perp2 = perp1.cross(normal);

	Vector3d p1(point);
	Vector3d p2(point);
	Vector3d p3(point);
	p2 = p2 + perp1;
	p3 = p3 + perp2;

	return Maths::calcPlaneValues(&p1, &p2, &p3);
}

Vector3d Maths::calcWorldSpaceDirectionVectorFromScreenSpaceCoords(float clickPosX, float clickPosY, int screenW, int screenH, float nearPlane, float vFOV, float camYaw, float camPitch)
{
    int displayWidth = screenW;
	int displayHeight = screenH;
    float aspectRatio = ((float)displayWidth)/displayHeight;

    float normalizedX = (clickPosX-(displayWidth/2))/((displayWidth/2));
    float normalizedY = (clickPosY-(displayHeight/2))/((displayHeight/2));

    float frustrumLengthY = nearPlane*tan(Maths::toRadians(vFOV/2));
    float frustrumLengthX = aspectRatio*frustrumLengthY;

    float cameraSpaceCoordX = normalizedX*frustrumLengthX;
    float cameraSpaceCoordY = -normalizedY*frustrumLengthY;
    float cameraSpaceCoordZ = -nearPlane;

    Vector3d cameraSpaceDirection(
            cameraSpaceCoordX,
            cameraSpaceCoordY,
            cameraSpaceCoordZ);
    cameraSpaceDirection.normalize();

    Vector3d xAxis(-1, 0, 0);
    Vector3d yAxis(0, -1, 0);

    Vector3d worldSpaceOffset = Maths::rotatePoint(&cameraSpaceDirection, &xAxis, Maths::toRadians(camPitch));

    return Maths::rotatePoint(&worldSpaceOffset, &yAxis, Maths::toRadians(camYaw));
}

//https://answers.unity.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
Vector2d Maths::interpolateUVTriangle(Triangle3D* tri, Vector3d* f)
{
    Vector3d p1(tri->p1.x, tri->p1.y, tri->p1.z);
    Vector3d p2(tri->p2.x, tri->p2.y, tri->p2.z);
    Vector3d p3(tri->p3.x, tri->p3.y, tri->p3.z);

    // calculate vectors from point f to vertices p1, p2 and p3:
    Vector3d f1 = p1-f;
    Vector3d f2 = p2-f;
    Vector3d f3 = p3-f;
    (p1-p2).cross(p1-p3);
    // calculate the areas (parameters order is essential in this case):
    Vector3d va  = (p1-p2).cross(p1-p3); // main triangle cross product
    Vector3d va1 = (f2).cross(f3); // p1's triangle cross product
    Vector3d va2 = (f3).cross(f1); // p2's triangle cross product
    Vector3d va3 = (f1).cross(f2); // p3's triangle cross product
    float a = va.length(); // main triangle area
    // calculate barycentric coordinates with sign:
    float a1 = va1.length()/a * Maths::sign(va.dot(va1));
    float a2 = va2.length()/a * Maths::sign(va.dot(va2));
    float a3 = va3.length()/a * Maths::sign(va.dot(va3));
    // find the uv corresponding to point f (uv1/uv2/uv3 are associated to p1/p2/p3):
    Vector2d uv = tri->uv1*a1 + tri->uv2*a2 + tri->uv3*a3;
    uv.y = -uv.y;
    return uv;
}
