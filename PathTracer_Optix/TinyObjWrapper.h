#pragma once



#include <util/tiny_obj_loader.h>
#include <string>
#include <vector>
#include <vector_types.h>

struct Vertex {
  float x, y, z;// , padding;
};

enum BSDFType {
  BSDF_DIFFUSE,
  BSDF_METALLIC,
  BSDF_REFRACTION,
};

struct Material {
  float3 diffuse;
  float3 emission;
  float roughness;
  float metallic;
  float ior;
  BSDFType bsdfType;
};


struct float3;

class TinyObjWrapper
{
  public:
    TinyObjWrapper() {};
    TinyObjWrapper(const std::string& filename);
    ~TinyObjWrapper() {};

    bool loadFile(const std::string& filename);



   
   

    std::vector<float> getVerticesFloat() const;

    std::vector<Material> getMaterials() const;
    std::vector<uint32_t> getMaterialIndices() const;
    std::vector<uint32_t> getIndexBuffer() const;

    size_t getNumMaterials() const;

private:
  bool dataLoaded = false;
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig reader_config;

  std::vector<float> _vertices;

  std::vector<Material> _materials;
  std::vector<uint32_t> _materialIndices;
  std::vector<uint32_t> _indexBuffer;

  void _updateVertices();
  void _updateMaterials();
  void _updateMaterialIndices();
  void _updateIndexBuffer();

};