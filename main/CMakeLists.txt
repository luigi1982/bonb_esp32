idf_component_register(SRCS "logmel.cpp" "main.cpp" "inference.cpp" "model_data.c"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES "esp-dsp" "esp-tflite-micro")