// Panting detection and context classification for the dog collar.
//
// Pure C, no hardware dependencies: feed it one frame every 10 ms (100 Hz)
// containing the accelerometer sample and the microphone loudness envelope,
// then call pd_update() once per second.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PD_FRAME_HZ 100 // frame rate of pd_push_frame()
#define PD_WIN      400 // 4 s analysis window

typedef enum {
    PD_ACT_REST,
    PD_ACT_WALK,
    PD_ACT_RUN,
} pd_activity_t;

typedef enum {
    PD_STATE_CALM,         // not panting
    PD_STATE_PLAY,         // panting while/after exercise, not hot: active and happy
    PD_STATE_COOLING,      // panting at rest in heat: regulating body temperature
    PD_STATE_HOT_EXERTION, // panting while/after exercise in heat: overheating risk
    PD_STATE_OTHER,        // panting at rest, not hot, no exercise: excitement, stress or pain
} pd_state_t;

typedef struct {
    float hot_c;             // ambient at or above this means heat panting is expected
    float danger_c;          // ambient at or above this raises a heat alert immediately
    float walk_rms_g;        // dynamic acceleration RMS separating rest from walking
    float run_rms_g;         // dynamic acceleration RMS separating walking from running
    uint32_t recovery_s;     // panting this long after activity still counts as exercise
    uint32_t other_alert_s;  // unexplained panting this long raises an alert
    uint32_t hot_ex_alert_s; // panting while exercising in heat this long raises an alert
    uint32_t cooling_alert_s;// heat panting at rest this long raises an alert
} pd_config_t;

typedef struct {
    pd_state_t state;
    pd_activity_t activity;
    bool panting;
    bool alert;           // owner should check on the dog
    bool ambient_known;
    float ambient_c;
    float pant_rate_bpm;  // breaths per minute, 0 when not panting
    float audio_score;    // 0..1 rhythm strength heard by the mic
    float imu_score;      // 0..1 rhythm strength felt by the IMU (rest only)
    float motion_rms_g;
    uint32_t state_s;     // seconds in the current state
} pd_result_t;

typedef struct {
    pd_config_t cfg;

    float acc[PD_WIN]; // high-passed acceleration magnitude, g
    float env[PD_WIN]; // high-passed audio envelope
    float env_raw[PD_WIN];
    int head;
    int filled;

    float grav;        // gravity estimate (low-pass of magnitude)
    float acc_hp_prev_in, acc_hp_prev_out;
    float env_hp_prev_in, env_hp_prev_out;

    float ambient_c;   // NAN when unknown

    int on_count, off_count;
    bool panting;
    uint32_t last_active_s;
    bool ever_active;
    pd_state_t state;
    uint32_t state_since_s;
} pd_ctx_t;

void pd_default_config(pd_config_t *cfg);
void pd_init(pd_ctx_t *ctx, const pd_config_t *cfg);

// ax/ay/az in g; audio_env is the RMS of the band-passed mic signal over the
// 10 ms frame, in any consistent unit.
void pd_push_frame(pd_ctx_t *ctx, float ax, float ay, float az, float audio_env);

// Ambient air temperature in °C, or NAN when unknown.
void pd_set_ambient(pd_ctx_t *ctx, float celsius);

// Call once per second. Returns false until the first window is full.
bool pd_update(pd_ctx_t *ctx, uint32_t now_s, pd_result_t *out);

const char *pd_state_name(pd_state_t s);

#ifdef __cplusplus
}
#endif
