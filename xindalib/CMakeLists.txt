cmake_minimum_required(VERSION 3.4.1)

add_library(rkssd4j
            SHARED
            src/main/cpp/Algorithm_jni.cpp
            src/main/cpp/mobilenet_eye.cpp
            src/main/cpp/direct_texture.cc)


find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

set(link_libs ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/librknn_api.so
              EGL
              GLESv2
              android
              ${log-lib}
)

target_link_libraries( # Specifies the target library.
                       rkssd4j
                       ${link_libs} )