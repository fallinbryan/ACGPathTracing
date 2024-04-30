#include "tinylogger.h"
#include "TinyObjWrapper.h"
#include "ConfigReader.h"
#include "AppController.h"

void context_log_callback(unsigned int level, const char* tag, const char* message, void* /*callbackdata */)
{

  std::map<unsigned int, tlog::ESeverity> level_map = {
    {1, tlog::ESeverity::Error},
    {2, tlog::ESeverity::Error},
    {3, tlog::ESeverity::Warning},
    {4, tlog::ESeverity::Info}
  };

  if (level_map.find(level) == level_map.end())
    level = 4;

  tlog::log(level_map[level]) << "[" << tag << "]:" << message;

}


int main() {
    tlog::debug() << "Starting Application...";

    std::string CONFIG_FILE = "AppConfig.cfg";
    
    tlog::debug() << "Reading Config File: " << CONFIG_FILE;
    ConfigReader config(CONFIG_FILE);
    std::string objfilepath = config.getString("INPUT OBJ FILE");
    
    tlog::debug() << "Reading OBJ File: " << objfilepath;
    TinyObjWrapper obj(objfilepath);
    
    OptixSettings optix_settings = {};
    

    optix_settings.logCallback = context_log_callback;
    optix_settings.scene = obj;
    optix_settings.scene_width = config.getInt("OUTPUT WIDTH");
    optix_settings.scene_height = config.getInt("OUTPUT HEIGHT");;
    optix_settings.samples_per_launch = config.getInt("SAMPLES PER PIXEL");
    optix_settings.max_depth = config.getInt("MAXIMUM RECRUSION DEPTH");
    optix_settings.useDirectLighting = true;
    optix_settings.useImportanceSampling = true;

    /*  
    *  NEW SECTION, MOVE THIS TO A DIFFERENT FILE ONCE IT IS WORKING
       -- loading the volume data from a file and putting onto the GPU
       -- then set a pointer to the volume data in the OptixSettings struct
    */
    // Load volume data from file
    
    std::string volume_file = config.getString("VOLUME FILE");
    std::ifstream volume_stream(volume_file, std::ios::binary);
    if (!volume_stream.is_open()) {
      tlog::error() << "Unable to open volume file: " << volume_file;
      return 1;
    }
    size_t volume_height, volume_width, volume_depth, voxel_sz;
    volume_height = config.getInt("VOLUME HEIGHT");
    volume_width = config.getInt("VOLUME WIDTH");
    volume_depth = config.getInt("VOLUME DEPTH");
    voxel_sz = sizeof(float) * 4;
    size_t volume_size = volume_height * volume_width * volume_depth;
    std::vector<float> h_volume_data(volume_size * voxel_sz);
    float* d_volume_data;
    volume_stream.read(reinterpret_cast<char*>(h_volume_data.data()), volume_size * voxel_sz);
    if (volume_stream.gcount() != volume_size * voxel_sz) {
      tlog::error() << "Unable to read volume data from file: " << volume_file;
      return 1;
    }
    volume_stream.close();
    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&d_volume_data, volume_size * voxel_sz));
    CUDA_CHECK(cudaMemcpy(d_volume_data, h_volume_data.data(), volume_size * voxel_sz, cudaMemcpyHostToDevice));
    optix_settings.volume_data = d_volume_data;
    optix_settings.volume_sz = volume_size * voxel_sz;
    optix_settings.volume_width = volume_width;
    optix_settings.volume_height = volume_height;
    optix_settings.volume_depth = volume_depth;

    //
 
    try {

      tlog::debug() << "Creating OptixManager...";
      OptixManager* optix_manager = AppController::getInstance().setOptixManager(optix_settings);

      tlog::debug() << "Initializing OptixManager...";
      if (!optix_manager->init()) {
        tlog::error() << "Unable to initialize Optix";
        return 1;
       }

      tlog::debug() << "Starting Render Loop...";
      optix_manager->render_loop();

      tlog::debug() << "Printing Benchmark...";
      optix_manager->print_benchmark();

    }
    catch (const std::exception& e) {
      tlog::error() << "Caught exception: " << e.what();
    }
    
    return 0;
}