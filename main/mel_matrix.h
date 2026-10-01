#pragma once

// Mel filterbank + STFT framing configuration.
//
// NOTE: this is a STANDARD mel filterbank generated with
// tf.signal.linear_to_mel_weight_matrix(num_mel_bins=64, num_spectrogram_bins=257,
// sample_rate=16000, lower_edge_hertz=125.0, upper_edge_hertz=7600.0).
// It was NOT exported from the original training pipeline (that file was missing from
// the project), so it reproduces the (400, 64) logmel shape the model expects but is
// not guaranteed to be bit-exact with training. Fine for a hardware inference smoke
// test with synthetic input; replace with the real exported matrix before relying on
// model accuracy with real audio.

#define LOGMEL_SAMPLE_RATE 16000
#define LOGMEL_FRAME_LENGTH 400
#define LOGMEL_FRAME_STEP 160
#define LOGMEL_FFT_LENGTH 512
#define LOGMEL_NUM_SPECTROGRAM_BINS 257
#define LOGMEL_NUM_MEL_BINS 64
