#include <math.h>
#include <cstdio>

#include "pixel.h"


void Pixel::set(double newX, double newY, double newZ)
{
	r = newX;
	g = newY;
	b = newZ;
}

void Pixel::set(Pixel* other)
{
	r = other->r;
	g = other->g;
	b = other->b;
}

void Pixel::scale(double scale)
{
	r *= scale;
	g *= scale;
	b *= scale;
}

Pixel Pixel::operator + (const Pixel &other)
{
	return Pixel(r + other.r, g + other.g, b + other.b);
}

Pixel::Pixel()
{
	r = 0;
	g = 0;
	b = 0;
}

Pixel::Pixel(double x, double y, double z)
{
	this->r = x;
	this->g = y;
	this->b = z;
}

Pixel::Pixel(Pixel* base)
{
	this->r = base->r;
	this->g = base->g;
	this->b = base->b;
}
