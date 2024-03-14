#pragma once



#include <util/tiny_obj_loader.h>
#include <string>
#include <vector>
#include <vector_types.h>

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
    std::vector<tinyobj::material_t> getMaterials() const;
    std::vector<uint32_t> getMaterialIndices() const;

    size_t getNumMaterials() const;

private:
  bool dataLoaded = false;
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig reader_config;

  std::vector<Vertex> _vertices;
  std::vector<tinyobj::material_t> _materials;
  std::vector<uint32_t> _materialIndices;

  void _updateVertices();
  void _updateMaterials();
  void _updateMaterialIndices();

  uint64_t floatToFixed(float value, float minValue, float maxValue, int bits);
  uint64_t mortonCode3D(uint64_t x, uint64_t y, uint64_t z);
  std::vector<uint64_t> computeMortonCodes();
  std::vector<size_t> computeSortedIndices();
  void reorderData();

};