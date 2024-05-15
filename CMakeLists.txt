
cmake_minimum_required (VERSION 3.5)

project (alogrithm)

# set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories (${PROJECT_SOURCE_DIR}/inc 
                    /home/xianmu/MyLib/openblas/include/)

find_library(OpenBlas_Lib openblas HINTS /home/xianmu/MyLib/openblas/lib/)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/alogrithmSrc   SRC_FILE)


# set (SRC_LIST ${PROJECT_SOURCE_DIR}/src/*)
add_executable (alogrithm ${PROJECT_SOURCE_DIR}/src/main.c ${SRC_FILE})
target_link_libraries (alogrithm ${OpenBlas_Lib} m)










