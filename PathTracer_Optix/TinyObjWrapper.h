#pragma once



#include <util/tiny_obj_loader.h>
#include <string>
#include <vector>

struct Vertex {
  float x, y, z, padding;
};

struct float3;

class TinyObjWrapper
{
  public:
    TinyObjWrapper() {};
    TinyObjWrapper(const std::string& filename);
    ~TinyObjWrapper() {};

    bool loadFile(const std::string& filename);



   
    std::vector<Vertex> getVertices() const;
    std::vector<float3> getMaterials() const;
    std::vector<uint32_t> getMaterialIndices() const;

    size_t getNumMaterials() const;

private:
  bool dataLoaded = false;
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig reader_config;

};

