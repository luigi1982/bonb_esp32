#include <iostream>
#include <cstdio>
#include <cinttypes>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"


static const char *TAG = "model_test";

// Tensor Arena - working memory for the interpreter
constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

void model_init(void)
{
    ESP_LOGI(TAG, "starting");
    ESP_LOGE(TAG, "free heap before allocate: %" PRIu32, heap_caps_get_free_size(MALLOC_CAP_8BIT));

    //load the model from the C array
    const tflite::Model *model = tflite::GetModel(q_model_tflite);

    // register all operations
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();

    //build the interpreter
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );

    // allocate tensors
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    } else {
        ESP_LOGE(TAG, "AllocateTensors() success");
    }

    // get input and output tensors
    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    //log tensor details
    ESP_LOGE(
        TAG,
        "input type: %d, shape: [%d, %d, %d, %d]",
        input->type, input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]
    );

    // get quantization parameters
    float input_scale = input->params.scale;
    int32_t input_zero_point = input->params.zero_point;
    float output_scale = output->params.scale;
    int32_t output_zero_point = output->params.zero_point;
}

float inference(float* logmel_spectrogram) {

    //change the code here so that logmel_spectrogram gets properly quantized
    //and set the input tensor of the model to be the result
    float* input_data = logmel_spectrogram / input_scale + input_zero_point;
    memset(input->input_data.int8, input_zero_point, input->bytes);

    // run inference - invoke interpreter and stop time
    uint32_t start = xTaskGetTickCount();
    TfLiteStatus invoke_status = interpreter.Invoke();
    uint32_t end = xTaskGetTickCount();
    uint32_t time_ms = (end - start) * portTICK_PERIOD_MS;
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return;
    }

    ESP_LOGE(TAG, "inference in ms: %" PRId32, time_ms);

    // dequantize the output
    int8_t logit_quantized = output->data.int8[0];
    float logit = (logit_quantized - output_zero_point) * output_scale;

    if (logit > 0) {
        ESP_LOGE(TAG, "logit value %f -> it's a doggo man", logit);
    } else {
        ESP_LOGE(TAG, "logit value %f -> that's no bark, c'mon", logit);
    }

    return logit;
}