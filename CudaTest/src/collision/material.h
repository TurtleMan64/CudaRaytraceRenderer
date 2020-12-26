#ifndef MATERIAL_H
#define MATERIAL_H

#include <string>

#include <SDL2/SDL.h>
#include "../toolbox/vector.h"

class Material
{
public:
    std::string name;

	SDL_Surface* texture;

    int type; //0 = diffuse. 1 = shiny

    Material(std::string name);

	Material(std::string name, SDL_Surface* texture, int type);
};

#endif
