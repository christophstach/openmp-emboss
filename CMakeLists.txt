cmake_minimum_required(VERSION 3.15)
project(openmp_emboss)

set(CMAKE_CXX_STANDARD 17)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/CMakeFiles/CPM.cmake")
set(CPM_VERSION 0.15.1)

if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://raw.githubusercontent.com/TheLartians/CPM.cmake/v${CPM_VERSION}/cmake/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})

include(${CPM_DOWNLOAD_LOCATION})

file(COPY "resources/images" DESTINATION "resources")

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

CPMFindPackage(
    NAME cxxopts
    GITHUB_REPOSITORY jarro2783/cxxopts
    VERSION 2.2.0
    GIT_TAG v2.2.0
)


add_executable(openmp_emboss main.cpp)


target_link_libraries(
        openmp_emboss
        PUBLIC OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
        cxxopts
)