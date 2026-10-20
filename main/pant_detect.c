#include "pant_detect.h"

#include <math.h>
#include <string.h>

// Dogs pant at roughly 90-420 breaths/min. Search 1.5-7 Hz.
#define LAG_MIN (PD_FRAME_HZ / 7)       // 14 frames
#define LAG_MAX (PD_FRAME_HZ * 2 / 3)   // 66 frames
#define ACT_WIN (PD_FRAME_HZ * 2)       // 2 s window for activity RMS

#define ON_SCORE_REST  0.45f
#define ON_SCORE_RUN   0.55f // gait adds rhythm of its own, demand a clearer signal
#define OFF_SCORE      0.30f
#define ON_COUNT       3     // consecutive seconds above threshold to switch on
#define OFF_COUNT      5     // consecutive seconds below threshold to switch off

// Breathing bursts make the loudness swing strongly; steady noise does not.
#define AUDIO_MIN_DEPTH 0.25f   // RMS of loudness swing / mean loudness
#define AUDIO_MIN_ENV   1e-4f   // ~-80 dBFS: below this the mic hears nothing
#define IMU_MIN_RMS_G   0.004f
#define GAIT_TOL        0.08f   // audio rhythm this close to stride rate is footfalls

// First-order high-pass coefficient for 0.5 Hz at 100 Hz: removes drift and
// posture changes but keeps the breathing rhythm.
#define HP_A 0.9695f
#define GRAV_A 0.01f

void pd_default_config(pd_config_t *cfg)
{
    cfg->hot_c = 26.0f;
    cfg->danger_c = 32.0f;
    cfg->walk_rms_g = 0.06f;
    cfg->run_rms_g = 0.35f;
    cfg->recovery_s = 300;
    cfg->other_alert_s = 180;
    cfg->hot_ex_alert_s = 300;
    cfg->cooling_alert_s = 1200;
}

void pd_init(pd_ctx_t *ctx, const pd_config_t *cfg)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->cfg = *cfg;
    ctx->grav = 1.0f;
    ctx->ambient_c = NAN;
    ctx->state = PD_STATE_CALM;
}

void pd_set_ambient(pd_ctx_t *ctx, float celsius)
{
    ctx->ambient_c = celsius;
}

void pd_push_frame(pd_ctx_t *ctx, float ax, float ay, float az, float audio_env)
{
    float mag = sqrtf(ax * ax + ay * ay + az * az);
    ctx->grav += GRAV_A * (mag - ctx->grav);
    float dyn = mag - ctx->grav;

    float acc_hp = HP_A * (ctx->acc_hp_prev_out + dyn - ctx->acc_hp_prev_in);
    ctx->acc_hp_prev_in = dyn;
    ctx->acc_hp_prev_out = acc_hp;

    float env_hp = HP_A * (ctx->env_hp_prev_out + audio_env - ctx->env_hp_prev_in);
    ctx->env_hp_prev_in = audio_env;
    ctx->env_hp_prev_out = env_hp;

    ctx->acc[ctx->head] = acc_hp;
    ctx->env[ctx->head] = env_hp;
    ctx->env_raw[ctx->head] = audio_env;
    ctx->head = (ctx->head + 1) % PD_WIN;
    if (ctx->filled < PD_WIN) {
        ctx->filled++;
    }
}

// Strength (0..1) and frequency of the strongest rhythm in 1.5-7 Hz, from the
// normalized autocorrelation of the last PD_WIN frames.
static float periodicity(const float *ring, int head, float *freq_hz)
{
    static float x[PD_WIN];
    float mean = 0.0f;
    for (int i = 0; i < PD_WIN; i++) {
        x[i] = ring[(head + i) % PD_WIN];
        mean += x[i];
    }
    mean /= PD_WIN;

    float r0 = 0.0f;
    for (int i = 0; i < PD_WIN; i++) {
        x[i] -= mean;
        r0 += x[i] * x[i];
    }
    *freq_hz = 0.0f;
    if (r0 < 1e-12f) {
        return 0.0f;
    }

    float r[LAG_MAX + 1];
    float best = 0.0f;
    for (int lag = LAG_MIN; lag <= LAG_MAX; lag++) {
        float s = 0.0f;
        for (int i = 0; i + lag < PD_WIN; i++) {
            s += x[i] * x[i + lag];
        }
        r[lag] = (s / (PD_WIN - lag)) / (r0 / PD_WIN);
        if (r[lag] > best) {
            best = r[lag];
        }
    }
    if (best <= 0.0f) {
        return 0.0f;
    }

    // A rhythm also peaks at 2x, 3x its period: take the shortest lag that is a
    // local maximum close to the best peak, which is the true breathing rate.
    for (int lag = LAG_MIN + 1; lag < LAG_MAX; lag++) {
        if (r[lag] >= 0.85f * best && r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1]) {
            *freq_hz = (float)PD_FRAME_HZ / lag;
            return r[lag] > 1.0f ? 1.0f : r[lag];
        }
    }
    return 0.0f; // best was at a range edge: no rhythm inside the band
}

static float window_rms(const float *ring, int head, int n)
{
    float s = 0.0f;
    for (int i = 1; i <= n; i++) {
        float v = ring[(head - i + PD_WIN) % PD_WIN];
        s += v * v;
    }
    return sqrtf(s / n);
}

static void set_state(pd_ctx_t *ctx, pd_state_t s, uint32_t now_s)
{
    if (ctx->state != s) {
        ctx->state = s;
        ctx->state_since_s = now_s;
    }
}

bool pd_update(pd_ctx_t *ctx, uint32_t now_s, pd_result_t *out)
{
    const pd_config_t *c = &ctx->cfg;
    if (ctx->filled < PD_WIN) {
        return false;
    }

    // Activity from the last 2 s of motion.
    float rms = window_rms(ctx->acc, ctx->head, ACT_WIN);
    pd_activity_t act = rms >= c->run_rms_g ? PD_ACT_RUN
                      : rms >= c->walk_rms_g ? PD_ACT_WALK
                      : PD_ACT_REST;
    if (act != PD_ACT_REST) {
        ctx->last_active_s = now_s;
        ctx->ever_active = true;
    }

    // Audio: panting is a loud, regular in/out rhythm.
    float env_mean = 0.0f;
    for (int i = 0; i < PD_WIN; i++) {
        env_mean += ctx->env_raw[i];
    }
    env_mean /= PD_WIN;
    float depth = env_mean > AUDIO_MIN_ENV
                ? window_rms(ctx->env, ctx->head, PD_WIN) / env_mean : 0.0f;
    float audio_hz;
    float audio_score = periodicity(ctx->env, ctx->head, &audio_hz);
    if (depth < AUDIO_MIN_DEPTH) {
        audio_score = 0.0f;
    }

    // While moving, footfalls and tag jingle tick at the step rate the IMU
    // sees. Discard an audio rhythm locked to it. A dog whose breathing is
    // locked 1:1 to its gallop is therefore only reported once it slows down.
    if (act != PD_ACT_REST && audio_score > 0.0f) {
        float gait_hz;
        if (periodicity(ctx->acc, ctx->head, &gait_hz) > 0.3f &&
            fabsf(audio_hz / gait_hz - 1.0f) < GAIT_TOL) {
            audio_score = 0.0f;
        }
    }

    // IMU: at rest the chest/throat movement of panting shows up at the collar.
    // While moving, gait swamps it, so only the mic is trusted.
    float imu_hz = 0.0f;
    float imu_score = 0.0f;
    if (act == PD_ACT_REST && window_rms(ctx->acc, ctx->head, PD_WIN) >= IMU_MIN_RMS_G) {
        imu_score = periodicity(ctx->acc, ctx->head, &imu_hz);
    }

    float score = audio_score > imu_score ? audio_score : imu_score;
    float rate_hz = audio_score >= imu_score ? audio_hz : imu_hz;

    float on_thr = act == PD_ACT_RUN ? ON_SCORE_RUN : ON_SCORE_REST;
    if (!ctx->panting) {
        ctx->on_count = score >= on_thr ? ctx->on_count + 1 : 0;
        if (ctx->on_count >= ON_COUNT) {
            ctx->panting = true;
            ctx->off_count = 0;
        }
    } else {
        ctx->off_count = score < OFF_SCORE ? ctx->off_count + 1 : 0;
        if (ctx->off_count >= OFF_COUNT) {
            ctx->panting = false;
            ctx->on_count = 0;
        }
    }

    // Context: why is the dog panting?
    bool ambient_known = !isnan(ctx->ambient_c);
    bool hot = ambient_known && ctx->ambient_c >= c->hot_c;
    bool danger = ambient_known && ctx->ambient_c >= c->danger_c;
    bool active = act != PD_ACT_REST ||
                  (ctx->ever_active && now_s - ctx->last_active_s < c->recovery_s);

    if (!ctx->panting) {
        set_state(ctx, PD_STATE_CALM, now_s);
    } else if (active && hot) {
        set_state(ctx, PD_STATE_HOT_EXERTION, now_s);
    } else if (active) {
        set_state(ctx, PD_STATE_PLAY, now_s);
    } else if (hot) {
        set_state(ctx, PD_STATE_COOLING, now_s);
    } else {
        set_state(ctx, PD_STATE_OTHER, now_s);
    }

    uint32_t in_state = now_s - ctx->state_since_s;
    bool alert = false;
    switch (ctx->state) {
    case PD_STATE_HOT_EXERTION: alert = danger || in_state >= c->hot_ex_alert_s; break;
    case PD_STATE_COOLING:      alert = danger || in_state >= c->cooling_alert_s; break;
    case PD_STATE_OTHER:        alert = in_state >= c->other_alert_s; break;
    default: break;
    }

    out->state = ctx->state;
    out->activity = act;
    out->panting = ctx->panting;
    out->alert = alert;
    out->ambient_known = ambient_known;
    out->ambient_c = ctx->ambient_c;
    out->pant_rate_bpm = ctx->panting ? rate_hz * 60.0f : 0.0f;
    out->audio_score = audio_score;
    out->imu_score = imu_score;
    out->motion_rms_g = rms;
    out->state_s = in_state;
    return true;
}

const char *pd_state_name(pd_state_t s)
{
    switch (s) {
    case PD_STATE_CALM:         return "calm";
    case PD_STATE_PLAY:         return "happy / exercising";
    case PD_STATE_COOLING:      return "cooling down (heat)";
    case PD_STATE_HOT_EXERTION: return "exercising in heat";
    case PD_STATE_OTHER:        return "panting without cause";
    }
    return "?";
}
