//#include <util/tiny_obj_loader.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "TinyObjWrapper.h"
#include <iostream>

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
#include <numeric>



TinyObjWrapper::TinyObjWrapper(const std::string& filename)
{
  loadFile(filename);
}


bool TinyObjWrapper::loadFile(const std::string& filename)
{
  std::string err;
  std::string warn;

  bool ret = reader.ParseFromFile(filename, reader_config);

  if (!ret) { 
    if(!reader.Error().empty()) std::cerr << "TinyObjReader: " << reader.Error();
   }
  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }
  dataLoaded = ret;

  if (ret)
  {
    _updateVertices();
    _updateMaterials();
    _updateMaterialIndices();
    _updateIndexBuffer();

  }

  return ret;
}

std::vector<float> TinyObjWrapper::getVerticesFloat() const
{
  return _vertices;
}

std::vector<tinyobj::material_t> TinyObjWrapper::getMaterials() const
{

  return _materials;
}

std::vector<uint32_t> TinyObjWrapper::getMaterialIndices() const
{

  return _materialIndices;
}

size_t TinyObjWrapper::getNumMaterials() const
{
  return _materials.size();
}


std::vector<uint32_t> TinyObjWrapper::getIndexBuffer() const
{
  return _indexBuffer;
}


void TinyObjWrapper::_updateMaterials()
{
  auto& materials = reader.GetMaterials();
  _materials = materials;
}

void TinyObjWrapper::_updateMaterialIndices()
{
  auto& shapes = reader.GetShapes();
  std::vector<uint32_t> materialIndices;
  if (dataLoaded)
  {
    for (auto& shape : shapes)
    {
      for (auto& index : shape.mesh.material_ids)
      {
        materialIndices.push_back(index);
      }
    }
  }
  _materialIndices = materialIndices;
}

void TinyObjWrapper::_updateVertices()
{
  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();
  

  
  if (dataLoaded)
  {
    _vertices = attrib.vertices;
 
  }
  
}

void TinyObjWrapper::_updateIndexBuffer()
{
  auto& shapes = reader.GetShapes();
  std::vector<uint32_t> indexBuffer;
  if (dataLoaded)
  {
    for (auto& shape : shapes)
    {
      for (auto& index : shape.mesh.indices)
      {
        indexBuffer.push_back(index.vertex_index);
      }
    }
  }
  _indexBuffer = indexBuffer;
}

