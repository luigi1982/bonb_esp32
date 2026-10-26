# ESP32-S3 Dog Bark Detector (full pipeline)

End-to-end audio classification on an ESP32-S3: an INMP441 I2S microphone feeds a log-mel front end, which feeds an int8 TensorFlow Lite Micro model (DS-CNN) that outputs the probability that the last ~4 seconds of audio contain a dog bark.

```
INMP441 mic ──I2S──► 4 s audio buffer ──► log-mel + z-norm ──► int8 quantize ──► TFLite Micro ──► p(dog)
 (16 kHz, 24-bit)    (63,920 samples)     (398 x 80 floats)                       (DS-CNN)
```

## Hardware

- ESP32-S3 with **octal PSRAM** (the model's activations and the audio/spectrogram buffers live in PSRAM)
- INMP441 I2S MEMS microphone

| INMP441 | ESP32-S3 |
|---------|----------|
| WS      | GPIO 16  |
| SCK     | GPIO 17  |
| SD      | GPIO 18  |
| L/R     | GND (left slot) |
| VDD/GND | 3V3 / GND |

Pins are set at the top of `main/main.cpp`.

## Software requirements

- ESP-IDF 5.x (target `esp32s3`)
- Managed components (fetched automatically on first build, see `main/idf_component.yml`):
  - `espressif/esp-dsp` – FFT and vector ops
  - `espressif/esp-tflite-micro` – TensorFlow Lite Micro
- Python + TensorFlow, only if you need to regenerate the mel filterbank (`tools/gen_mel_sparse.py`)

A `.devcontainer/` (ESP-IDF + QEMU) and VS Code settings are included.

## Build and run

```sh
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

Make sure PSRAM is enabled (`CONFIG_SPIRAM=y`, octal mode); the committed `sdkconfig` already has this.

Output, one line per 4-second window:

```
frontend ms=...  inference ms=...  p(dog)=0.123 |#####                                   |
```

The bar is 40 characters wide and proportional to `p(dog)`.

## How it works

### Audio capture (`main/main.cpp`)
Reads 32-bit I2S slots at 16 kHz (mono, left). The 24 valid bits are shifted down and scaled to [-1, 1]. 398 frames × 160-sample hop + 400-sample window = 63,920 samples (~4 s) are collected per inference.

### Front end (`main/logmel.cpp`)
Designed to match `tf.signal.stft` + `tf.signal.linear_to_mel_weight_matrix` from training:

- 25 ms frames (400 samples), 10 ms hop (160), 512-point FFT, periodic Hann window, no end padding
- 80 mel bins, 60–7600 Hz (parameters in `main/mel_matrix.h`)
- `log(mel + 1e-6)`, then z-normalization using the mean/std of the whole spectrogram
- Optimizations: two real frames are packed into one complex FFT, and the mel matrix is stored sparsely (`main/mel_matrix_sparse_col.h`)

### Inference (`main/inference.cpp`)
- Model is embedded as a C array (`main/model_data.c`, derived from `ds_cnn_s__logmel_features_int8.tflite`)
- Input `[1, 398, 80, 1]`-style int8 tensor, quantized with the model's scale/zero-point
- Ops registered: Conv2D, DepthwiseConv2D, FullyConnected, Mean, Softmax
- Output is already softmaxed: index 0 = dog, index 1 = no-dog
- 1280 KB tensor arena, allocated from PSRAM (falls back to internal RAM, which will not fit this model)

## Binary size

This pipeline is meant to be integrated into another firmware where flash is tight, so the app binary was trimmed. Measured on an ESP32-S3 (sizes of `build/full_pipeline.bin`, timings averaged over ~12 windows, first window dropped as warm-up):

| Version | Commit | Binary size | Δ vs. baseline | Front end | Inference |
|---|---|---|---|---|---|
| Baseline | `f608fe4` | 395,536 B | – | 216.97 ms | 2472.9 ms |
| + FFT without esp-dsp init tables | `b62c5f2` | 379,168 B | −16,368 B | 216.05 ms | 2472.9 ms |
| + Strip TFLite error strings | `fbd5604` | 363,008 B | −32,528 B | 216.08 ms | 2468.3 ms |
| + TFLite at `-Os` (reverted) | `ef46eca` | 350,656 B | −44,880 B | 216.08 ms | 2942.0 ms |
| + esp-nn at `-Os` | `186ceec` | 346,928 B | −48,608 B | 216.14 ms | 2955.7 ms |
| **Current: all of the above except TFLite `-Os`** | `b43b7ec` | **359,296 B** | **−36,240 B (−9.2 %)** | 216.04 ms | 2482.4 ms |

What was done:

1. **FFT setup without `dsps_fft2r_init_fc32`** (`main/logmel.cpp`). esp-dsp's init references bit-reversal tables for every FFT size from 16 to 4096 points (~16 KB), so the linker keeps all of them, even though `dsps_bit_rev_fc32` never reads them. The front end now builds the twiddle table itself (`dsps_gen_w_r2_fc32` + `dsps_bit_rev_fc32_ansi`, the same steps as the init) and calls the ESP32-S3 assembly FFT `dsps_fft2r_fc32_aes3_` directly. Costs 2 KB of static RAM for the table and saves 960 B of heap. This ties the code to the ESP32-S3; a `#error` catches other targets.
2. **`TF_LITE_STRIP_ERROR_STRINGS`** (top-level `CMakeLists.txt`). Removes TFLite Micro's error messages, source file paths and the code that prints them. TFLite failures now return only a status code (the `ESP_LOGE` messages in `inference.cpp` remain). Remove this define when debugging TFLite problems.
3. **esp-nn compiled with `-Os`** instead of its hard-coded `-O2` (top-level `CMakeLists.txt`). Saves ~3.7 KB for ~14 ms (~0.5 %) more inference time.
4. **Tried and reverted: TFLite Micro with `-Os`** instead of its hard-coded `-O3`. Saved another ~12 KB but made inference ~19 % slower.

The compiler settings are set in this project's top-level `CMakeLists.txt` (after `project()`), not in `managed_components/`, which the component manager can overwrite. **When integrating into another firmware, copy that block into its top-level `CMakeLists.txt`**, or the savings are lost.

Not yet applied, and better decided in the target firmware's `sdkconfig`: the project still builds with Debug optimization (`-Og`) and assertions enabled. Most of the remaining ~46 KB of string data comes from ESP-IDF itself (log messages, assert text, `esp_err_to_name` tables), so it is controlled by the host firmware's configuration.

## Project layout

```
main/
  main.cpp                   mic setup, capture loop, console output
  logmel.cpp/.h              log-mel front end
  mel_matrix.h               STFT/mel configuration constants
  mel_matrix_sparse_col.h    generated sparse mel filterbank
  inference.cpp/.h           TFLite Micro setup, quantization, invoke
  model_data.c/.h            model weights as a C array
  norm_stats.h, norm_stats.json   per-bin stats from an earlier model (unused; the current front end normalizes per spectrogram)
tools/gen_mel_sparse.py      regenerates mel_matrix_sparse_col.h
ds_cnn_s__logmel_features_int8.tflite   source model
```

## Regenerating the mel filterbank

If you change the mel parameters, they must match the training front end exactly. Update `main/mel_matrix.h` and run:

```sh
python tools/gen_mel_sparse.py --fft-length 512 --num-mel-bins 80 \
    --sample-rate 16000 --lower-edge 60 --upper-edge 7600
```

Note: the script's defaults (64 bins, 125 Hz lower edge) differ from the current configuration, so pass the flags above.

## Status / known issues

- Inference takes roughly 2.5 s per window with the arena in PSRAM, so the loop runs slower than real time per 4 s window plus front end time.
- Earlier models misclassified human speech as barking; the current model and per-spectrogram normalization are still being evaluated.
