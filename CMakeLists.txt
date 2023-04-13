cmake_minimum_required(VERSION 3.15)
project(tflite-example CXX)

find_package(tensorflowlite REQUIRED CONFIG)
find_package(OpenCV REQUIRED CONFIG)

add_executable(tflite-example src/main.cpp)

target_link_libraries(tflite-example PRIVATE tensorflow::tensorflowlite opencv::opencv)
target_compile_features(tflite-example PRIVATE cxx_std_17)

install(TARGETS tflite-example DESTINATION "."
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        BUNDLE DESTINATION bin)

file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/assets/*.*")

foreach(asset ${ASSETS})
        add_custom_command(TARGET tflite-example POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${asset}
                $<TARGET_FILE_DIR:tflite-example>)
endforeach(asset)