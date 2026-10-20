#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_check.h"
#include <esp_timer.h>

#include "logmel.h"
#include "inference.h"
#include "pant_audio.h"
#include "freertos/stream_buffer.h"

static const char *TAG = "mic_test";

// Wiring (INMP441 -> ESP32-S3)
static constexpr gpio_num_t I2S_WS_PIN  = GPIO_NUM_16;  // WS  (word select / left-right clock)
static constexpr gpio_num_t I2S_SCK_PIN = GPIO_NUM_17;  // SCK (bit clock)
static constexpr gpio_num_t I2S_SD_PIN  = GPIO_NUM_18;  // SD  (serial data, mic -> ESP32)

static constexpr uint32_t SAMPLE_RATE_HZ = 16000;       // 16 kHz is plenty for speech
// The model input is 398 frames x 80 mel bins: 398 = 1 + (N - 400) / 160  ->  N = 63920 (~4 s)
static constexpr int      NUM_FRAMES     = 398;
static constexpr size_t   FRAME_SAMPLES  = LOGMEL_FRAME_LENGTH + (NUM_FRAMES - 1) * LOGMEL_FRAME_STEP;
static constexpr size_t   READ_CHUNK     = 1024;          // samples per I2S read

static i2s_chan_handle_t rx_chan = nullptr;
// The mic must be read continuously for the panting detector, even while the
// bark model is busy (~2.5 s). A reader task owns the I2S channel, feeds the
// panting detector, and hands the samples to the bark pipeline via this buffer.
static StreamBufferHandle_t audio_stream = nullptr;
static constexpr size_t STREAM_BYTES = 16384 * sizeof(int32_t);  // ~1 s of audio

static void mic_init()
{
    // 1) Create an I2S "RX" (receive) channel. We only listen, we never send audio out.
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, nullptr, &rx_chan));

    // 2) Configure it as standard I2S (Philips format), which is what the INMP441 speaks.
    i2s_std_config_t std_cfg = {};
    std_cfg.clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ);
    // The mic sends 24 valid bits inside a 32-bit slot, so read 32-bit slots.
    // MONO + LEFT: L/R pin is tied to GND, so the mic talks in the "left" time slot.
    std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;            // INMP441 has no master clock pin
    std_cfg.gpio_cfg.bclk = I2S_SCK_PIN;
    std_cfg.gpio_cfg.ws   = I2S_WS_PIN;
    std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;            // we don't output audio
    std_cfg.gpio_cfg.din  = I2S_SD_PIN;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
}

static void mic_reader_task(void *)
{
    static int32_t buf[READ_CHUNK];
    while (true) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(rx_chan, buf, sizeof(buf), &bytes_read, portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "i2s read failed: %s", esp_err_to_name(err));
            pant_audio_reset();
            continue;
        }
        pant_audio_feed(buf, bytes_read / sizeof(int32_t));
        // Never block: if the bark pipeline is busy the buffer fills and audio is dropped for it.
        xStreamBufferSend(audio_stream, buf, bytes_read, 0);
    }
}

extern "C" void app_main(void)
{
    if (!model_init()) {
        ESP_LOGE(TAG, "model_init() failed, stopping");
        return;
    }

    mic_init();
    pant_audio_init();
    audio_stream = xStreamBufferCreate(STREAM_BYTES, sizeof(int32_t));
    if (!audio_stream) {
        ESP_LOGE(TAG, "audio stream buffer allocation failed");
        return;
    }
    xTaskCreate(mic_reader_task, "mic_reader", 4096, nullptr, 6, nullptr);
    ESP_LOGI(TAG, "Microphone running at %u Hz. Make some noise!", (unsigned)SAMPLE_RATE_HZ);

    // ~250 KB of audio + ~125 KB of spectrogram: prefer PSRAM, fall back to internal RAM.
    static int32_t raw[READ_CHUNK];
    int num_frames = 0;
    float* samples = (float*)heap_caps_malloc(FRAME_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!samples) samples = (float*)heap_caps_malloc(FRAME_SAMPLES * sizeof(float), MALLOC_CAP_8BIT);
    float* out = (float*)heap_caps_malloc(NUM_FRAMES * LOGMEL_NUM_MEL_BINS * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!out) out = (float*)heap_caps_malloc(NUM_FRAMES * LOGMEL_NUM_MEL_BINS * sizeof(float), MALLOC_CAP_8BIT);
    if (!samples || !out) {
        ESP_LOGE(TAG, "out of memory for audio / spectrogram buffers");
        return;
    }

    while (true) {

        // Collect FRAME_SAMPLES samples from the mic, READ_CHUNK at a time.
        size_t filled = 0;
        xStreamBufferReset(audio_stream);   // discard audio from while the model was running
        while (filled < FRAME_SAMPLES) {
            size_t want = FRAME_SAMPLES - filled;
            if (want > READ_CHUNK) want = READ_CHUNK;

            size_t bytes_read = 0;
            bytes_read = xStreamBufferReceive(audio_stream, raw, want * sizeof(int32_t), portMAX_DELAY);

            // The 24-bit sample sits in the top 24 bits of each 32-bit word -> shift down by 8,
            // then scale to -1..1 for the log-mel computation.
            size_t n = bytes_read / sizeof(int32_t);
            for (size_t i = 0; i < n; i++) {
                int32_t s = raw[i] >> 8;                // signed 24-bit: -8388608 .. +8388607
                samples[filled + i] = s / 8388608.0f;
            }
            filled += n;
        }


        // compute logmel spectrogram
        int64_t t0f = esp_timer_get_time();
        logmel_compute(samples, FRAME_SAMPLES, out, &num_frames);
        int64_t t1f = esp_timer_get_time();
        float frontend_ms = (t1f - t0f) / 1000.0f;

        // pass spectrogram to model and complete forward pass
        int64_t t0i = esp_timer_get_time();
        float p_dog = inference(out, num_frames);
        int64_t t1i = esp_timer_get_time();
        float inference_ms = (t1i - t0i) / 1000.0f;

        if (std::isnan(p_dog)) {
            ESP_LOGE(TAG, "inference failed");
            continue;
        }

        // Draw a bar (0..40 chars) proportional to the probability of "dog".
        int bar = (int)lroundf(p_dog * 40.0f);
        if (bar > 40) bar = 40;
        if (bar < 0) bar = 0;
        char bar_str[41];
        for (int i = 0; i < 40; i++) bar_str[i] = (i < bar) ? '#' : ' ';
        bar_str[40] = '\0';

        printf("frontend ms=%5.2f  inference ms=%5.2f  p(dog)=%5.3f |%s|\n",
            frontend_ms, inference_ms, p_dog, bar_str
        );
    }
}
