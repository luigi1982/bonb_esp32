#pragma once

#include <stdint.h>
#include "mel_matrix.h"

// Computes a log-mel spectrogram from raw float audio samples, matching
// TensorFlow's tf.signal.stft (periodic Hann window, pad_end=False) +
// tf.signal.linear_to_mel_weight_matrix + log(x + 1e-6), using the fixed
// mel filterbank exported from the training pipeline (mel_matrix.h).
//
// The result is z-normalized with the mean/std of the spectrogram itself
// (computed over all frames and mel bins), as the model expects.
//
// Combined optimization version with:
// - Precomputed Hann window (avoids redundant trigonometric calculations)
// - Sparse mel matrix column representation (reduces memory bandwidth in mel multiplication)
//
// num_frames = 1 + (num_samples - LOGMEL_FRAME_LENGTH) / LOGMEL_FRAME_STEP
// out must hold num_frames * LOGMEL_NUM_MEL_BINS floats.
void logmel_compute(const float* samples, int num_samples, float* out, int* out_num_frames);

int logmel_num_frames(int num_samples);
