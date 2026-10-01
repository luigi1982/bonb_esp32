#include "inference.h"

#include <cinttypes>
#include <cmath>
#include "esp_log.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "logmel.h"


static const char *TAG = "model_test";

// Tensor Arena - working memory for the interpreter
constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// state shared between model_init() and inference()
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor *s_input = nullptr;
static TfLiteTensor *s_output = nullptr;

bool model_init(void)
{
    ESP_LOGI(TAG, "free heap before allocate: %" PRIu32, heap_caps_get_free_size(MALLOC_CAP_8BIT));

    //load the model from the C array
    const tflite::Model *model = tflite::GetModel(q_model_new_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema version %" PRIu32 " != supported %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // register all operations
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();

    //build the interpreter (static so it outlives this function)
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );

    // allocate tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }
    ESP_LOGI(TAG, "AllocateTensors() success, arena used: %u / %d bytes",
             (unsigned)interpreter.arena_used_bytes(), kTensorArenaSize);

    // get input and output tensors
    s_input = interpreter.input(0);
    s_output = interpreter.output(0);

    if (s_input->type != kTfLiteInt8 || s_output->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "expected int8 input/output, got input type %d, output type %d",
                 s_input->type, s_output->type);
        return false;
    }

    //log tensor details
    TfLiteIntArray *dims = s_input->dims;
    ESP_LOGI(TAG, "input shape (%d dims): [%d, %d, %d, %d], scale %f, zero point %d",
             dims->size,
             dims->size > 0 ? dims->data[0] : 0, dims->size > 1 ? dims->data[1] : 0,
             dims->size > 2 ? dims->data[2] : 0, dims->size > 3 ? dims->data[3] : 0,
             s_input->params.scale, (int)s_input->params.zero_point);
    ESP_LOGI(TAG, "output scale %f, zero point %d",
             s_output->params.scale, (int)s_output->params.zero_point);

    s_interpreter = &interpreter;
    return true;
}

float inference(const float* logmel_spectrogram, int num_frames)
{
    if (s_interpreter == nullptr) {
        ESP_LOGE(TAG, "inference() called before successful model_init()");
        return NAN;
    }

    // the spectrogram must exactly fill the input tensor (1 byte per int8 element)
    const size_t num_values = (size_t)num_frames * LOGMEL_NUM_MEL_BINS;
    if (num_values != s_input->bytes) {
        ESP_LOGE(TAG, "spectrogram has %u values (%d frames), model expects %u",
                 (unsigned)num_values, num_frames, (unsigned)s_input->bytes);
        return NAN;
    }

    // quantize: q = round(x / scale) + zero_point, clamped to int8 range
    const float inv_scale = 1.0f / s_input->params.scale;
    const int32_t input_zero_point = s_input->params.zero_point;
    int8_t *input_data = s_input->data.int8;
    for (size_t i = 0; i < num_values; i++) {
        int32_t q = (int32_t)lroundf(logmel_spectrogram[i] * inv_scale) + input_zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input_data[i] = (int8_t)q;
    }

    // run inference - invoke interpreter and stop time
    TfLiteStatus invoke_status = s_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return NAN;
    }

    // dequantize the output
    int8_t logit_quantized = s_output->data.int8[0];
    float logit = (logit_quantized - s_output->params.zero_point) * s_output->params.scale;

    return logit;
}
