#include "common.h"
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


void main() {
   
    std::string CONFIG_FILE = "AppConfig.cfg";
    ConfigReader config(CONFIG_FILE);
    std::string objfilepath = config.getString("INPUT OBJ FILE");
    OptixSettings optix_settings = {};
    
    optix_settings.logCallback = context_log_callback;
    optix_settings.scene = TinyObjWrapper(objfilepath);
    optix_settings.scene_width = config.getInt("OUTPUT WIDTH");
    optix_settings.scene_height = config.getInt("OUTPUT HEIGHT");;
    optix_settings.samples_per_launch = config.getInt("SAMPLES PER PIXEL");
    optix_settings.max_depth = config.getInt("MAXIMUM RECRUSION DEPTH");
    optix_settings.useDirectLighting = true;
    optix_settings.useImportanceSampling = true;


    OptixManager* optix_manager = AppController::getInstance().setOptixManager(optix_settings);
 
    try {

      optix_manager->init();

      optix_manager->render_loop();

      optix_manager->print_benchmark();

    }
    catch (const std::exception& e) {
      tlog::error() << "Caught exception: " << e.what();
    }
    
}