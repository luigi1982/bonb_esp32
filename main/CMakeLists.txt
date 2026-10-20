idf_component_register(SRCS "model_data.c" "logmel.cpp" "main.cpp" "inference.cpp" "pant_audio.cpp" "pant_detect.c"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES "esp-dsp" "esp-tflite-micro")