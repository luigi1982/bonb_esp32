#pragma once

// Loads the TFLite model from model_data.c, builds the interpreter and
// allocates its tensors. Must be called once before inference().
// Returns false if setup failed.
bool model_init(void);

// Quantizes a log-mel spectrogram (num_frames * LOGMEL_NUM_MEL_BINS floats,
// frame-major as produced by logmel_compute), copies it into the model's
// input tensor, runs the model and returns the dequantized output logit.
// Returns NAN on error (model not initialized, size mismatch, invoke failure).
float inference(const float* logmel_spectrogram, int num_frames);
