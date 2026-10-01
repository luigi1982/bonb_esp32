#pragma once

// Mel filterbank + STFT framing configuration.
//
// Mel filterbank config, matching the training front end:
// tf.signal.linear_to_mel_weight_matrix(num_mel_bins=40, num_spectrogram_bins=257,
// sample_rate=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0).
// Regenerate mel_matrix_sparse_col.h with tools/gen_mel_sparse.py when changing these.

#define LOGMEL_SAMPLE_RATE 16000
#define LOGMEL_FRAME_LENGTH 480
#define LOGMEL_FRAME_STEP 160
#define LOGMEL_FFT_LENGTH 512
#define LOGMEL_NUM_SPECTROGRAM_BINS 257
#define LOGMEL_NUM_MEL_BINS 40
