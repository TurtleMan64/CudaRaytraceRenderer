#include <SDL2/SDL.h>
#include <string>
#include "material.h"

Material::Material(std::string name)
{
    this->name = name;
    this->texture = nullptr;
    this->type = 0;
}

Material::Material(std::string name, SDL_Surface* texture, int type)
{
    this->name = name;
	this->texture = texture;
    this->type = type;
}
