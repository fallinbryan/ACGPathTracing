#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

OPTIX_add_sample_executable(optixPathTracer target_name
PathTracerMain.cpp
pathTracerPrograms.cu
common.h
common.cpp
AppController.h
AppController.cpp
ConfigReader.h
ConfigReader.cpp
GLFCallbacks.h
GLFCallbacks.cpp
OptixManager.h
OptixManager.cpp
TinyObjWrapper.h
TinyObjWrapper.cpp
pathTracer.h
  OPTIONS -rdc true
)

target_link_libraries(${target_name}
  ${CUDA_LIBRARIES}
)

# Copy AppConfig.cfg to the output directory after building the target
add_custom_command(
    TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/AppConfig.cfg
            $<TARGET_FILE_DIR:${target_name}>
    COMMENT "Copying AppConfig.cfg to output directory"
)

# Visual Studio integration for AppConfig.cfg
set(APPCONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/AppConfig.cfg)
source_group("Resources" FILES ${APPCONFIG_FILE})
add_custom_target(AppConfigResource ALL
    SOURCES ${APPCONFIG_FILE}
    COMMENT "Adding AppConfig.cfg to Visual Studio Solution Explorer under Resources"
)
