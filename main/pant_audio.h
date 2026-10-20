#pragma once

#include <cstddef>
#include <cstdint>

// Panting detector fed from the same I2S microphone as the bark classifier.
// Call pant_audio_init() once, then pant_audio_feed() with every block read
// from the mic (24-bit samples in the top bits of 32-bit words, mono).
// Audio must arrive continuously: the rhythm detector needs gap-free 4 s windows.
void pant_audio_init(void);
void pant_audio_feed(const int32_t *raw, size_t n);
// Drop the current window after an audio gap.
void pant_audio_reset(void);
