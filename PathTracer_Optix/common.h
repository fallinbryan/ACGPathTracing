

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif 

#ifndef OPTIX_MAN_ONLY
  #ifndef OPT_COMMON_H
    #define OPT_COMMON_H
    #ifndef NO_GLFW_INCLUDE
      #include <GLFW/glfw3.h>
      #include <glad/glad.h>
    #endif
    #ifndef NO_OPTIX_DEFINES
      #include <optix.h>
      #include <optix_function_table_definition.h>
      #include <optix_stubs.h>
      #include <optix_stack_size.h>
    #endif
  #endif
#endif // OPTIX_MAN_ONLY