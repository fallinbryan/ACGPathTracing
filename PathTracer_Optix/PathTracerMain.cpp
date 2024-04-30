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