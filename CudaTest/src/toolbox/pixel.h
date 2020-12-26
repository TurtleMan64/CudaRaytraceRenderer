#ifndef PIXEL_H
#define PIXEL_H

#include <math.h>


class Pixel
{
public:
	double r;
	double g;
	double b;

	Pixel();
	Pixel(double r, double g, double b);
	Pixel(Pixel* other);

	void set(double x, double y, double z);

	void set(Pixel* other);

	void scale(double scale);

	Pixel operator + (const Pixel &other);
};

#endif
