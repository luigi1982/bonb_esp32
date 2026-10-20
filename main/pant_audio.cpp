#include "pant_audio.h"

#include <cmath>
#include <cstdio>

#include "esp_log.h"
#include "esp_timer.h"

#include "pant_detect.h"

static const char *TAG = "pant";

static constexpr float    FS_HZ       = 16000.0f;                // must match SAMPLE_RATE_HZ in main.cpp
static constexpr int      BLOCK       = (int)(FS_HZ / PD_FRAME_HZ);  // samples per 10 ms frame
static constexpr float    HP_A        = 1.0f / (1.0f + 2.0f * 3.14159265f * 300.0f / FS_HZ);  // 300 Hz high-pass
static constexpr int      STARTUP_BLK = 50;                      // let the mic settle for 0.5 s

static pd_ctx_t s_pd;
static float    s_hp_in, s_hp_out, s_sumsq;
static int      s_count;
static int      s_startup;
static int      s_frames;     // frames since last pd_update()
static uint32_t s_now_s;

void pant_audio_reset(void)
{
    pd_config_t cfg;
    pd_default_config(&cfg);
    pd_init(&s_pd, &cfg);    // ambient unknown, no recent exercise
    s_hp_in = s_hp_out = s_sumsq = 0.0f;
    s_count = 0;
    s_frames = 0;
    s_startup = STARTUP_BLK;
}

void pant_audio_init(void)
{
    s_now_s = 0;
    pant_audio_reset();
}

void pant_audio_feed(const int32_t *raw, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (s_startup > 0) {
            if (++s_count >= BLOCK) { s_count = 0; s_startup--; }
            continue;
        }
        const float x = (float)(raw[i] >> 8) / 8388608.0f;
        const float y = HP_A * (s_hp_out + x - s_hp_in);
        s_hp_in = x;
        s_hp_out = y;
        s_sumsq += y * y;
        if (++s_count < BLOCK) continue;

        // One 10 ms frame: loudness envelope in, motion fixed at rest (1 g, no IMU on this board).
        pd_push_frame(&s_pd, 0.0f, 0.0f, 1.0f, sqrtf(s_sumsq / BLOCK));
        s_sumsq = 0.0f;
        s_count = 0;

        if (++s_frames >= PD_FRAME_HZ) {
            s_frames = 0;
            s_now_s++;
            pd_result_t r;
            if (pd_update(&s_pd, s_now_s, &r)) {
                ESP_LOGI(TAG, "panting=%d alert=%d rate=%.0f/min audio=%.2f state=%s",
                         r.panting, r.alert, (double)r.pant_rate_bpm, (double)r.audio_score,
                         pd_state_name(r.state));
            }
        }
    }
}
