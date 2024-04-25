/*
 * TinyObjWrapper.h
 *
 * This header file defines the TinyObjWrapper class, which is a wrapper around the tinyobj library.
 * The tinyobj library is used for loading .obj files, which are a common format for 3D model data.
 * The TinyObjWrapper class provides a more convenient and high-level interface for loading these files,
 * and it also provides additional functionality such as storing the loaded data in various formats (e.g., as a vector of floats),
 * and keeping track of materials and indices associated with the 3D model.
 *
 * The file also defines several structs (Vertex, Material, BSDFType) that are used to represent the data in the .obj files.
 */


#pragma once



#include <util/tiny_obj_loader.h>
#include <string>
#include <vector>
#include <vector_types.h>
#include <optix.h>

struct Vertex {
  float x, y, z;// , padding;
};

enum BSDFType {
  BSDF_DIFFUSE,
  BSDF_METALLIC,
  BSDF_REFRACTION,
  BSDF_VOLUME,
};

struct Material {
  float3 diffuse;
  float3 emission;
  float volumeDensity;
  float roughness;
  float metallic;
  float ior;
  BSDFType bsdfType;
};


struct float3;


/**
 * @class TinyObjWrapper
 *
 * @brief A wrapper class for the tinyobj library.
 *
 * This class provides a high-level interface for loading .obj files using the tinyobj library.
 * It also provides additional functionality such as storing the loaded data in various formats,
 * and keeping track of materials and indices associated with the 3D model.
 *
 * @method TinyObjWrapper() Default constructor.
 * @method TinyObjWrapper(const std::string& filename) Overloaded constructor that takes a filename to load.
 * @method ~TinyObjWrapper() Default destructor.
 * @method bool loadFile(const std::string& filename) Loads an .obj file.
 * @method std::vector<float> getVerticesFloat() const Returns the vertices as a vector of floats.
 * @method std::vector<Material> getMaterials() const Returns the materials associated with the 3D model.
 * @method std::vector<uint32_t> getMaterialIndices() const Returns the material indices associated with the 3D model.
 * @method std::vector<uint32_t> getIndexBuffer() const Returns the index buffer associated with the 3D model.
 * @method size_t getNumMaterials() const Returns the number of materials associated with the 3D model.
 *
 * @member dataLoaded A boolean indicating whether data has been loaded.
 * @member reader An instance of the tinyobj::ObjReader class for reading .obj files.
 * @member reader_config An instance of the tinyobj::ObjReaderConfig class for configuring the reader.
 * @member _vertices A vector of floats representing the vertices of the 3D model.
 * @member _materials A vector of Material structs representing the materials of the 3D model.
 * @member _materialIndices A vector of uint32_t representing the material indices of the 3D model.
 * @member _indexBuffer A vector of uint32_t representing the index buffer of the 3D model.
 */
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
    std::vector<OptixAabb> getAabbs() const;

    size_t getNumMaterials() const;

private:
  bool dataLoaded = false;
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig reader_config;

  std::vector<float> _vertices;

  std::vector<Material> _materials;
  std::vector<uint32_t> _materialIndices;
  std::vector<uint32_t> _indexBuffer;

  std::vector<OptixAabb> _aabbs;

  void _updateVertices();
  void _updateMaterials();
  void _updateMaterialIndices();
  void _updateIndexBuffer();
  void _updateAabbs();

};