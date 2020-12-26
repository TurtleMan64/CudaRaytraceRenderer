#include <SDL2/SDL.h> 
#include <SDL2/SDL_image.h>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <vector>
#include <list>

#include "objLoader.h"
#include "../toolbox/vector.h"
#include "../toolbox/split.h"
#include "../toolbox/getline.h"
#include "collisionmodel.h"
#include "triangle3d.h"
#include "material.h"

CollisionModel* loadCollisionModel(std::string filePath, std::string fileName)
{
	CollisionModel* collisionModel = new CollisionModel;

	std::list<Material*> materials;

	Material* currMaterial = nullptr;

	std::ifstream file(filePath + fileName);
	if (!file.is_open())
	{
		std::fprintf(stdout, "Error: Cannot load file '%s'\n", (filePath + fileName).c_str());
		file.close();
		return collisionModel;
	}

	std::string line;

	std::vector<Vector3d> vertices;
    std::vector<Vector2d> uvs;


	while (!file.eof())
	{
		getlineSafe(file, line);

		char lineBuf[256];
		memcpy(lineBuf, line.c_str(), line.size()+1);

		int splitLength = 0;
		char** lineSplit = split(lineBuf, ' ', &splitLength);

		if (splitLength > 0)
		{
			if (strcmp(lineSplit[0], "v") == 0)
			{
				Vector3d vertex;
				vertex.x = std::stof(lineSplit[1]);
				vertex.y = std::stof(lineSplit[2]);
				vertex.z = std::stof(lineSplit[3]);
				vertices.push_back(vertex);
			}
            else if (strcmp(lineSplit[0], "vt") == 0)
			{
				Vector2d uv;
				uv.x = std::stof(lineSplit[1]);
				uv.y = std::stof(lineSplit[2]);
				uvs.push_back(uv);
			}
			else if (strcmp(lineSplit[0], "f") == 0)
			{
				int len = 0;
				char** vertex1 = split(lineSplit[1], '/', &len);
				char** vertex2 = split(lineSplit[2], '/', &len);
				char** vertex3 = split(lineSplit[3], '/', &len);

				Vector3d* vert1 = &vertices[std::stoi(vertex1[0]) - 1];
				Vector3d* vert2 = &vertices[std::stoi(vertex2[0]) - 1];
				Vector3d* vert3 = &vertices[std::stoi(vertex3[0]) - 1];

                Vector2d* uv1 = &uvs[std::stoi(vertex1[1]) - 1];
				Vector2d* uv2 = &uvs[std::stoi(vertex2[1]) - 1];
				Vector2d* uv3 = &uvs[std::stoi(vertex3[1]) - 1];

				Triangle3D* tri = new Triangle3D(vert1, vert2, vert3);
                tri->material = currMaterial;
                tri->uv1.set(uv1);
                tri->uv2.set(uv2);
                tri->uv3.set(uv3);

				collisionModel->triangles.push_back(tri);

				free(vertex1);
				free(vertex2);
				free(vertex3);
			}
			else if (strcmp(lineSplit[0], "usemtl") == 0)
			{
				currMaterial = nullptr;

				for (Material* mat : materials)
				{
					if (mat->name == lineSplit[1])
					{
						currMaterial = mat;
					}
				}
			}
			else if (strcmp(lineSplit[0], "mtllib") == 0)
			{
				std::ifstream fileMTL(filePath + lineSplit[1]);
				if (!fileMTL.is_open())
				{
					std::fprintf(stdout, "Error: Cannot load mtl file '%s'\n", (filePath + lineSplit[1]).c_str());
					fileMTL.close();
					file.close();
					return collisionModel;
				}

				std::string lineMTL;

				while (!fileMTL.eof())
				{
					getlineSafe(fileMTL, lineMTL);

					char lineBufMTL[256];
					memcpy(lineBufMTL, lineMTL.c_str(), lineMTL.size()+1);

					int splitLengthMTL = 0;
					char** lineSplitMTL = split(lineBufMTL, ' ', &splitLengthMTL);

					if (splitLengthMTL > 1)
					{
						if (strcmp(lineSplitMTL[0], "newmtl") == 0)
						{
							Material* newMaterial = new Material(lineSplitMTL[1]);
							materials.push_back(newMaterial);
						}
						else if (strcmp(lineSplitMTL[0], "type")   == 0 ||
								 strcmp(lineSplitMTL[0], "\ttype") == 0)
						{
							if (strcmp(lineSplitMTL[1], "mirror") == 0)
							{
								materials.back()->type = 1;
							}
						}
                        else if (strcmp(lineSplitMTL[0], "map_Kd")   == 0 ||
								 strcmp(lineSplitMTL[0], "\tmap_Kd") == 0)
						{
							SDL_Surface* tex = IMG_Load((filePath + lineSplitMTL[1]).c_str());
                            if (tex == nullptr)
                            {
                                std::fprintf(stdout, "Error: Cannot load texture '%s'\n", (filePath + lineSplitMTL[1]).c_str());
                            }
							materials.back()->texture = tex;
						}
					}
					free(lineSplitMTL);
				}
				fileMTL.close();
			}
		}
		free(lineSplit);
	}
	file.close();

	return collisionModel;
}
