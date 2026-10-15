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

static const char *TAG = "mic_test";

// Wiring (INMP441 -> ESP32-S3)
static constexpr gpio_num_t I2S_WS_PIN  = GPIO_NUM_16;  // WS  (word select / left-right clock)
static constexpr gpio_num_t I2S_SCK_PIN = GPIO_NUM_17;  // SCK (bit clock)
static constexpr gpio_num_t I2S_SD_PIN  = GPIO_NUM_18;  // SD  (serial data, mic -> ESP32)

static constexpr uint32_t SAMPLE_RATE_HZ = 16000;       // 16 kHz is plenty for speech
static constexpr size_t   FRAME_SAMPLES  = 8320;         // samples processed per loop (~520 ms)

static i2s_chan_handle_t rx_chan = nullptr;

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

extern "C" void app_main(void)
{
    if (!model_init()) {
        ESP_LOGE(TAG, "model_init() failed, stopping");
        return;
    }

    mic_init();
    ESP_LOGI(TAG, "Microphone running at %u Hz. Make some noise!", (unsigned)SAMPLE_RATE_HZ);

    static int32_t raw[FRAME_SAMPLES];
    int num_frames = 0;
    float* samples = (float*)heap_caps_malloc(FRAME_SAMPLES * sizeof(float), MALLOC_CAP_8BIT);
    float* out = (float*)heap_caps_malloc(150 * LOGMEL_NUM_MEL_BINS * sizeof(float), MALLOC_CAP_8BIT);

    while (true) {
        
        // Block until a full frame of samples has arrived from the mic.
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(rx_chan, raw, sizeof(raw), &bytes_read, portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "i2s read failed: %s", esp_err_to_name(err));
            continue;
        }

        size_t n = bytes_read / sizeof(int32_t);
        if (n == 0) continue;

        // The 24-bit sample sits in the top 24 bits of each 32-bit word -> shift down by 8.
        // Then measure loudness: peak = largest |sample|, avg = mean |sample|.

        for (size_t i = 0; i < n; i++) {
            
            int32_t s = raw[i] >> 8;                    // signed 24-bit: -8388608 .. +8388607
            // fill samples array for computation of logmel spectrogram
            samples[i] = s / 8388608.0f;

        }


        // compute logmel spectrogram
        int64_t t0f = esp_timer_get_time();
        logmel_compute(samples, FRAME_SAMPLES, out, &num_frames);
        int64_t t1f = esp_timer_get_time();
        float frontend_ms = (t1f - t0f) / 1000.0f;

        // pass spectrogram to model and complete forward pass
        int64_t t0i = esp_timer_get_time();
        float logit = inference(out, num_frames);
        int64_t t1i = esp_timer_get_time();
        float inference_ms = (t1i - t0i) / 1000.0f;

        if (std::isnan(logit)) {
            ESP_LOGE(TAG, "inference failed");
        }

        // Draw a simple bar (0..40 chars) from the average level. Speech is usually
        // well below full scale, so we scale so ~1/64 of full scale fills the bar.
        int bar = (int)(2*(40/(1 + exp(-logit)) - 20));
        if (bar > 40) bar = 40;
        char bar_str[41];
        for (int i = 0; i < 40; i++) bar_str[i] = (i < bar) ? '#' : ' ';
        bar_str[40] = '\0';

        printf("frontend ms=%5.2f  inference ms=%5.2f  logit=%+8.2f |%s|\n",
            frontend_ms, inference_ms, logit, bar_str
        );

        vTaskDelay(pdMS_TO_TICKS(520));                  // print ~2 lines/second
    }
}
