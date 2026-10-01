#include "logmel.h"
#include "mel_matrix_sparse_col.h"
#include <math.h>
#include <string.h>
#include <esp_dsp.h>

namespace {

constexpr int kFrameLength = LOGMEL_FRAME_LENGTH;
constexpr int kFrameStep = LOGMEL_FRAME_STEP;
constexpr int kFftLength = LOGMEL_FFT_LENGTH;
constexpr int kNumSpectrogramBins = LOGMEL_NUM_SPECTROGRAM_BINS;
constexpr int kNumMelBins = LOGMEL_NUM_MEL_BINS;

// Precomputed Hann window
float g_hann_window[kFrameLength];
bool g_hann_init = false;

bool g_fft_init = false;

void init_hann_window() {
  // NOTE: esp_dsp's dsps_wind_hann_f32 implements the symmetric Hann window
  // (divided by N-1), not the periodic Hann window TensorFlow's tf.signal.stft
  // uses by default (divided by N). Keep this manual periodic version so the
  // windowing matches the mel filterbank this pipeline was trained against.
  for (int j = 0; j < kFrameLength; j++) {
    g_hann_window[j] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)j / (float)kFrameLength);
  }
  g_hann_init = true;
}

// Copies samples[start..start+kFrameLength) into frame_buf (zero-padding past
// num_samples) and applies the Hann window in place.
void fill_windowed_frame(const float* samples, int num_samples, int start, float* frame_buf) {
  int valid = num_samples - start;
  if (valid > kFrameLength) valid = kFrameLength;
  if (valid < 0) valid = 0;
  memcpy(frame_buf, samples + start, valid * sizeof(float));
  if (valid < kFrameLength) {
    memset(frame_buf + valid, 0, (kFrameLength - valid) * sizeof(float));
  }
  dsps_mul_f32(frame_buf, g_hann_window, frame_buf, kFrameLength, 1, 1, 1);
}

void mel_from_magnitude(const float* magnitude, float* mel_out) {
  for (int m = 0; m < kNumMelBins; m++) {
    float acc = 0.0f;
    uint16_t row_start = g_mel_matrix_sparse_col_offsets[m];
    uint16_t row_len = g_mel_matrix_sparse_col_lengths[m];
    for (uint16_t i = 0; i < row_len; i++) {
      const MelMatrixEntry& entry = g_mel_matrix_sparse_cols[row_start + i];
      acc += magnitude[entry.row_idx] * entry.value;
    }
    mel_out[m] = acc; //logf(acc + 1e-6f);
  }
}

}  // namespace

int logmel_num_frames(int num_samples) {
  return (num_samples + kFrameStep - 1) / kFrameStep;
}

void logmel_compute(const float* samples, int num_samples, float* out, int* out_num_frames) {
  if (!g_hann_init) {
    init_hann_window();
  }

  if (!g_fft_init) {
    ESP_ERROR_CHECK(dsps_fft2r_init_fc32(NULL, kFftLength));
    g_fft_init = true;
  }

  const int num_frames = logmel_num_frames(num_samples);
  *out_num_frames = num_frames;

  // Interleaved complex buffer: fft_buf[2*j] = real, fft_buf[2*j+1] = imag.
  static float fft_buf[2 * kFftLength];
  static float frame_a[kFrameLength];
  static float frame_b[kFrameLength];
  static float magnitude_a[kNumSpectrogramBins];
  static float magnitude_b[kNumSpectrogramBins];

  int frame = 0;

  // Process frames in pairs: pack two real frames into one complex FFT
  // (frame_a -> real part, frame_b -> imaginary part) and unpack the two
  // real spectra with dsps_cplx2reC_fc32, halving the number of FFTs run.
  // This only reconstructs bins 0..kFftLength/2-1 (no Nyquist bin), which is
  // fine here since this mel filterbank never weights bin kFftLength/2.
  for (; frame + 1 < num_frames; frame += 2) {
    fill_windowed_frame(samples, num_samples, frame * kFrameStep, frame_a);
    fill_windowed_frame(samples, num_samples, (frame + 1) * kFrameStep, frame_b);

    memset(fft_buf, 0, sizeof(fft_buf));
    for (int j = 0; j < kFrameLength; j++) {
      fft_buf[2 * j] = frame_a[j];
      fft_buf[2 * j + 1] = frame_b[j];
    }

    dsps_fft2r_fc32(fft_buf, kFftLength);
    dsps_bit_rev_fc32(fft_buf, kFftLength);
    dsps_cplx2reC_fc32(fft_buf, kFftLength);

    const float* spec_a = fft_buf;
    const float* spec_b = fft_buf + kFftLength;
    for (int b = 0; b < kFftLength / 2; b++) {
      magnitude_a[b] = sqrtf(spec_a[2 * b] * spec_a[2 * b] + spec_a[2 * b + 1] * spec_a[2 * b + 1]);
      magnitude_b[b] = sqrtf(spec_b[2 * b] * spec_b[2 * b] + spec_b[2 * b + 1] * spec_b[2 * b + 1]);
    }

    mel_from_magnitude(magnitude_a, out + frame * kNumMelBins);
    mel_from_magnitude(magnitude_b, out + (frame + 1) * kNumMelBins);
  }

  // Trailing odd frame (if num_frames is odd): fall back to a plain
  // real-input complex FFT so we get the full kNumSpectrogramBins, including
  // the Nyquist bin.
  if (frame < num_frames) {
    fill_windowed_frame(samples, num_samples, frame * kFrameStep, frame_a);

    memset(fft_buf, 0, sizeof(fft_buf));
    for (int j = 0; j < kFrameLength; j++) {
      fft_buf[2 * j] = frame_a[j];
    }

    dsps_fft2r_fc32(fft_buf, kFftLength);
    dsps_bit_rev_fc32(fft_buf, kFftLength);

    for (int b = 0; b < kNumSpectrogramBins; b++) {
      const float re = fft_buf[2 * b];
      const float im = fft_buf[2 * b + 1];
      magnitude_a[b] = sqrtf(re * re + im * im);
    }

    mel_from_magnitude(magnitude_a, out + frame * kNumMelBins);
  }
}
