idf_component_register(SRCS "model_data.c" "logmel.cpp" "main.cpp" "inference.cpp"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES "esp-dsp" "esp-tflite-micro")