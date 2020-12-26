#ifndef OBJLOADER_H
#define OBJLOADER_H

class CollisionModel;

#include <string>

CollisionModel* loadCollisionModel(std::string filePath, std::string fileName);

#endif
