cmake_minimum_required(VERSION 3.16)
project(RawProcessing)

set(CMAKE_CXX_STANDARD 14)

add_library( # Sets the name of the library.
        main

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        main.cpp raw/CfaPattern.h raw/Demosaic.cpp raw/Demosaic.h)

add_library(libraw
        SHARED
        IMPORTED)

set_target_properties( # Specifies the target library.
        libraw

        # Specifies the parameter you want to define.
        PROPERTIES IMPORTED_LOCATION

        # Provides the path to the library you want to import.
        /home/sergeyfitis/dev/env/libraw.so)

add_library(libopencv_java430
        SHARED
        IMPORTED raw/Demosaic.cpp raw/Demosaic.h)


set_target_properties( # Specifies the target library.
        libopencv_java430

        # Specifies the parameter you want to define.
        PROPERTIES IMPORTED_LOCATION

        # Provides the path to the library you want to import.
        /home/sergeyfitis/dev/cpp-projects/opencv-4.3.0/build/lib/libopencv_java430.so)


add_executable(RawProcessing main.cpp raw/CfaPattern.h raw/Demosaic.cpp raw/Demosaic.h)
target_link_libraries(RawProcessing libraw libopencv_java430 )

