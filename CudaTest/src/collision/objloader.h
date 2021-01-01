#ifndef OBJLOADER_H
#define OBJLOADER_H

class CollisionModel;

#include <string>

SDL_Surface* loadSdlImage(const char* filepath);

CollisionModel* loadCollisionModel(std::string filePath, std::string fileName);

#endif
