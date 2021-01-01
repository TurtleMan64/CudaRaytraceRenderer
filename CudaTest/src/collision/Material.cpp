#include <SDL2/SDL.h>
#include <string>
#include "material.h"

Material::Material(std::string name)
{
    this->name = name;
    this->textureDiffuse = nullptr;
    this->textureNormal = nullptr;
    this->type = 0;
}

Material::Material(std::string name, SDL_Surface* textureDiffuse, SDL_Surface* textureNormal, int type)
{
    this->name = name;
	this->textureDiffuse = textureDiffuse;
    this->textureNormal = textureNormal;
    this->type = type;
}
