cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0054 NEW)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{NO_USE_VCPKG})
    if(DEFINED ENV{CI})
        message(STATUS "CI build detected.")
        set(VCPKG_INSTALLATION_ROOT ${CMAKE_CURRENT_LIST_DIR}/vcpkg)
        message(
            STATUS
            "Setting VCPKG_INSTALLATION_ROOT to ${CMAKE_CURRENT_LIST_DIR}/vcpkg"
        )
    elseif(DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_INSTALLATION_ROOT $ENV{VCPKG_ROOT})
        message(STATUS "VCPKG_INSTALLATION_ROOT is ${VCPKG_INSTALLATION_ROOT}")
    else()
        # get the vcpkg commit id from vcpkg.json
        file(READ ${CMAKE_CURRENT_LIST_DIR}/vcpkg.json VCPKG_JSON_STRING)
        string(JSON VCPKG_COMMIT_ID GET ${VCPKG_JSON_STRING} builtin-baseline)
        message(
            STATUS
            "Downloading vcpkg from https://github.com/microsoft/vcpkg/"
        )
        message(STATUS "Commit SHA512: ${VCPKG_COMMIT_ID}")
        include(FetchContent)
        FetchContent_Declare(
            vcpkg
            GIT_REPOSITORY https://github.com/microsoft/vcpkg/
            GIT_TAG ${VCPKG_COMMIT_ID}
        )
        FetchContent_MakeAvailable(vcpkg)
        message(STATUS "Downloading vcpkg -- done")
        set(VCPKG_INSTALLATION_ROOT ${vcpkg_SOURCE_DIR})
        message(STATUS "Bootstrapping vcpkg")
        if(WIN32)
            execute_process(
                COMMAND ${VCPKG_INSTALLATION_ROOT}/bootstrap-vcpkg.bat
            )
        elseif(UNIX)
            execute_process(
                COMMAND ${VCPKG_INSTALLATION_ROOT}/bootstrap-vcpkg.sh
            )
        endif()
    endif()
    if(EXISTS ${VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake)
        message(
            STATUS
            "Using Vcpkg toolchain: ${VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
        )
        set(CMAKE_TOOLCHAIN_FILE
            "${VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING
            ""
        )
        set(USING_VCPKG TRUE)
    else()
        message(
            FATAL_ERROR
            "Did not find vcpkg at specified location: ${VCPKG_INSTALLATION_ROOT}"
        )
    endif()
else()
    message(STATUS "Proceeding without vcpkg.")
endif()

project(Periodica LANGUAGES CXX)

find_package(Eigen3 CONFIG REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3 headers in: ${EIGEN3_INCLUDE_DIR}")
endif()
find_package(CGAL CONFIG REQUIRED)
if(DEFINED CGAL_MODULES_DIR AND Eigen3_FOUND)
    include(${CGAL_MODULES_DIR}/CGAL_Eigen3_support.cmake)
    message(STATUS "CGAL has Eigen3 support.")
endif()
find_package(GUDHI CONFIG REQUIRED)
if(GUDHI_FOUND)
    message(STATUS "Found Gudhi headers in: ${GUDHI_INCLUDE_DIR}")
endif()

# add_executable(Periodica debug.cpp delaunay.cpp)
# target_link_libraries(Periodica Eigen3::Eigen CGAL::CGAL Gudhi::Gudhi)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create the module
add_library(periodica MODULE auxiliary.cpp delaunay.cpp merge_tree.cpp persistence_image.cpp periodica.cpp)
target_link_libraries(periodica PRIVATE pybind11::module Eigen3::Eigen CGAL::CGAL Gudhi::Gudhi)

