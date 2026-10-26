# For more information about build system see
# https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html
# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(full_pipeline)

# Flash-size tweaks for esp-tflite-micro (set here, not in managed_components/,
# which the component manager may overwrite).
idf_component_get_property(tflite_lib espressif__esp-tflite-micro COMPONENT_LIB)
# Drop TFLite's error messages / source paths (MicroPrintf becomes a no-op).
# PUBLIC so every component including the TFLite headers agrees on it.
target_compile_definitions(${tflite_lib} PUBLIC TF_LITE_STRIP_ERROR_STRINGS)
# Note: TFLite itself stays at its hard-coded -O3. -Os there saved ~12 KB but
# made inference ~19% slower on the ESP32-S3.

# esp-nn hard-codes -O2; appended later on the command line, -Os wins.
# Saves ~3.7 KB for ~0.5% slower inference.
idf_component_get_property(esp_nn_lib espressif__esp-nn COMPONENT_LIB)
target_compile_options(${esp_nn_lib} PRIVATE -Os)
