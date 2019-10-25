cmake_minimum_required(VERSION 3.15)
project(openmp_emboss)

set(CMAKE_CXX_STANDARD 17)

file(COPY "resources/images" DESTINATION "resources")

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)



add_executable(openmp_emboss main.cpp)


target_link_libraries(
        openmp_emboss
        PUBLIC OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
)