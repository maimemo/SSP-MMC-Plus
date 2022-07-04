cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(SSP_MMC)

function(download_and_extract url name)
    if (EXISTS ${CMAKE_SOURCE_DIR}/${name})
        message("-- Found ${name}")
    else ()
        message("-- Downloading ${name}")
        file(DOWNLOAD ${url} ${CMAKE_SOURCE_DIR}/${name}.zip)
        message("-- Downloading ${name} - done")
        file(ARCHIVE_EXTRACT INPUT ${CMAKE_SOURCE_DIR}/${name}.zip DESTINATION ${CMAKE_SOURCE_DIR})
        file(REMOVE ${CMAKE_SOURCE_DIR}/${name}.zip)
    endif ()
endfunction()

download_and_extract(https://files.pythonhosted.org/packages/7b/91/89bbe2316b93671b6bccec094df6bc66109cf6d21a364cd2f1becd11ba3c/torch-1.10.2-cp39-none-macosx_11_0_arm64.whl torch)
set(CMAKE_PREFIX_PATH torch)
find_package(Torch REQUIRED)
find_package(range-v3 CONFIG REQUIRED)


include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(SSP_MMC DHP.cpp)
target_link_libraries(SSP_MMC PRIVATE range-v3 range-v3-meta range-v3::meta range-v3-concepts "${TORCH_LIBRARIES}" cnpy)
set_property(TARGET SSP_MMC PROPERTY CXX_STANDARD 14)
set_target_properties(SSP_MMC PROPERTIES LINK_FLAGS -Wl,-stack_size,0x20000000)