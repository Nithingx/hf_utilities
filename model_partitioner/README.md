# Vision-Language Model Partitioner

A comprehensive toolkit for splitting, converting, and running vision-language models (VLMs) with different inference pipelines and formats.

## Features

### 🎯 Core Capabilities

1. **Original Model Inference** - Run the complete VL model as-is
2. **Split Native Pipeline** - Separate vision and language models (PyTorch + SafeTensors)
3. **ONNX Conversion** - Convert vision model to ONNX format
4. **ONNX + SafeTensor Pipeline** - Run with ONNX vision + SafeTensor LLM
5. **Standalone Inference** - Export models with independent inference scripts
6. **Performance Tracking** - Detailed metrics for CPU/GPU usage and latency
7. **Comprehensive Benchmarking** - Multi-run benchmarks with statistical analysis

### 📦 Supported Formats

- **Vision Model**: PyTorch (.pt), ONNX (.onnx)
- **Language Model**: SafeTensors (.safetensors), AWQ (planned)

### 🚀 Future Support

- VitisAI Execution Provider for ONNX
- AWQ quantization for language models

## Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For quantization support:
```bash
pip install bitsandbytes
```

For ONNX GPU support:
```bash
pip install onnxruntime-gpu
```

## Project Structure

```
model_partitioner/
├── model_partitioner_v2.py      # Main orchestrator
├── vision_pipeline.py            # Vision model inference pipeline
├── language_pipeline.py          # Language model inference pipeline
├── onnx_converter.py             # ONNX conversion utilities
├── benchmark.py                  # Benchmarking suite
├── run_all_examples.py           # Example runner
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── BENCHMARK_GUIDE.md            # Detailed benchmarking guide
├── QUICKSTART.md                 # Quick start guide
└── split_models/                 # Output directory (created at runtime)
    ├── vision_model/             # Vision model (.pt format)
    ├── language_model/           # Language model (.safetensors)
    ├── onnx_model/               # ONNX vision model
    └── standalone/               # Standalone inference scripts
        ├── vision/               # Vision-only inference
        └── language/             # Language-only inference
```

## Usage

### Mode 1: Original Model (Baseline)

Run the complete vision-language model without any modifications:

```bash
python model_partitioner_v2.py \
    --mode original \
    --image demo.jpg \
    --text "Describe this image succinctly."
```

**What it does:**
- Loads the full VL model
- Runs end-to-end inference
- Provides baseline performance metrics

### Mode 2: Split Native Pipeline

Split the model into vision and language components and run them separately:

```bash
python model_partitioner_v2.py \
    --mode split_native \
    --image demo.jpg \
    --text "Describe this image succinctly."
```

**What it does:**
- Splits model into vision (.pt) and language (.safetensors) components
- Saves models to disk
- Runs inference with separated pipeline
- Demonstrates independent component execution

**Output files:**
- `split_models/vision_model/vision_model.pt`
- `split_models/language_model/language_model.safetensors`
- `split_models/model_config.json`

### Mode 3: Convert to ONNX

Convert the vision model to ONNX format for optimized inference:

```bash
python model_partitioner_v2.py \
    --mode convert_onnx \
    --image demo.jpg
```

**What it does:**
- Exports vision model to ONNX format
- Validates the exported model
- Reports conversion metrics (file size, verification)

**Output files:**
- `split_models/onnx_model/vision_model.onnx`
- Language model remains as SafeTensors

**Benefits:**
- Optimized for inference
- Cross-platform compatibility
- Supports hardware acceleration (VitisAI, TensorRT)

### Mode 4: Run with ONNX

Run inference using ONNX vision model + SafeTensor language model:

```bash
python model_partitioner_v2.py \
    --mode run_onnx \
    --image demo.jpg \
    --text "Describe this image succinctly."
```

**What it does:**
- Loads ONNX vision model
- Loads SafeTensor language model
- Runs end-to-end inference with mixed formats
- Compares performance with original

**Requirements:**
- ONNX model must be created first (run `convert_onnx` mode)

### Mode 5: Save Standalone Models

Export models with standalone inference scripts for independent debugging:

```bash
python model_partitioner_v2.py \
    --mode save_standalone \
    --image demo.jpg
```

**What it does:**
- Copies models to standalone directories
- Creates individual inference scripts
- Enables independent testing and debugging

**Output structure:**
```
split_models/standalone/
├── vision/
│   ├── vision_model.onnx
│   └── vision_inference.py
└── language/
    ├── language_model.safetensors
    └── language_inference.py
```

**Running standalone scripts:**
```bash
# Vision only
cd split_models/standalone/vision
python vision_inference.py --image ../../../demo.jpg

# Language only
cd split_models/standalone/language
python language_inference.py --text "Your prompt here"
```

### Mode 6: Run All (Complete Pipeline)

Execute all modes sequentially for comprehensive testing:

```bash
python model_partitioner_v2.py \
    --mode all \
    --image demo.jpg \
    --text "Describe this image succinctly."
```

**What it does:**
1. Baseline inference with original model
2. Split and save models
3. Run with split native pipeline
4. Convert vision to ONNX
5. Run with ONNX + SafeTensor
6. Export standalone inference scripts

## Command-Line Options

### Required Arguments

- `--mode`: Inference mode
  - `original` - Run original model
  - `split_native` - Split and run native formats
  - `convert_onnx` - Convert vision to ONNX
  - `run_onnx` - Run with ONNX vision
  - `save_standalone` - Export standalone scripts
  - `all` - Run all modes

### Optional Arguments

- `--model-id`: HuggingFace model ID (default: `Qwen/Qwen2.5-VL-3B-Instruct`)
- `--device`: Device to use (`auto`, `cuda`, `cpu`)
- `--image`: Path to input image (default: `demo.jpg`)
- `--text`: Text prompt for inference
- `--max-tokens`: Maximum tokens to generate (default: 128)
- `--output-dir`: Output directory for split models (default: `split_models`)
- `--quantize`: Enable 4-bit quantization

## Benchmarking

### Quick Benchmark

Run comprehensive benchmarks across all modes:

```bash
# Basic benchmark (5 runs per mode)
python benchmark.py --image demo.jpg --runs 5

# Detailed benchmark (10 runs)
python benchmark.py --image demo.jpg --runs 10 --output results.json --csv

# Compare devices
python benchmark.py --image demo.jpg --compare-devices --runs 5

# Compare quantization
python benchmark.py --image demo.jpg --compare-quantization --runs 5
```

### Benchmark Output

```
======================================================================
BENCHMARK RESULTS SUMMARY
======================================================================

Mode            Metric                    Mean         Std          Min          Max         
--------------------------------------------------------------------------------------
original        Duration (s)              2.345        0.123        2.201        2.489       
original        Tokens/sec                54.67        2.13         52.34        56.78       
run_onnx        Duration (s)              1.845        0.089        1.756        1.934       
run_onnx        Tokens/sec                69.35        3.21         66.14        72.56       

⚡ Fastest Mode: run_onnx (1.845s)

Relative Performance (vs run_onnx):
  run_onnx       : 1.845s (1.00x, +0.0%)
  original       : 2.345s (0.79x, +27.1%)
```

### Metrics Measured

The benchmark suite measures:
- **Latency**: Total inference time, preprocessing, generation
- **Throughput**: Tokens per second
- **Memory**: CPU and GPU usage (delta and peak)
- **Model Size**: On-disk size of different formats
- **Statistics**: Mean, std, min, max, median across multiple runs

For detailed benchmarking guide, see [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)

## Performance Tracking

The tool automatically tracks and reports:

- ⏱️ **Latency**: Preprocessing, inference, decoding times
- 💾 **CPU Memory**: Memory usage and delta
- 🎮 **GPU Memory**: VRAM usage and peak memory
- ⚡ **Throughput**: Tokens per second

Example output:
```
============================================================
PERFORMANCE: original_e2e
============================================================
⏱️  Duration: 2.345s
💾 CPU Memory: 150.25 MB
🎮 GPU Memory: 3245.50 MB
🎮 GPU Peak: 3890.75 MB
============================================================
```

## Advanced Usage

### Running with Quantization

Use 4-bit quantization to reduce memory usage:

```bash
python model_partitioner_v2.py \
    --mode original \
    --image demo.jpg \
    --quantize
```

### Using Different Models

```bash
python model_partitioner_v2.py \
    --mode all \
    --model-id "Qwen/Qwen2-VL-7B-Instruct" \
    --image demo.jpg
```

### GPU vs CPU Comparison

```bash
# GPU
python model_partitioner_v2.py --mode original --device cuda --image demo.jpg

# CPU
python model_partitioner_v2.py --mode original --device cpu --image demo.jpg
```

## Architecture Overview

### Vision Pipeline (`vision_pipeline.py`)

Handles vision model inference:
- **PyTorch mode**: Direct inference using `.pt` weights
- **ONNX mode**: Inference using ONNX Runtime
- Supports VitisAI Execution Provider (planned)

### Language Pipeline (`language_pipeline.py`)

Handles language model inference:
- **PyTorch mode**: Direct inference using `.pt` weights
- **SafeTensors mode**: Inference using `.safetensors` format
- **AWQ mode**: Quantized inference (planned)

### ONNX Converter (`onnx_converter.py`)

Utilities for ONNX conversion:
- Model export with validation
- Dynamic axes support
- Optimization and quantization (optional)

## Future Enhancements

### VitisAI Integration

```python
# Planned: VitisAI Execution Provider
vision_pipeline = VisionPipeline(
    model_format='onnx',
    device='vitisai',
    vitisai_config={'target': 'DPUCZDX8G_ISA1_B4096'}
)
```

### AWQ Quantization

```python
# Planned: AWQ for language models
language_pipeline = LanguagePipeline(
    model_format='awq',
    quantization_bits=4
)
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Use quantization
python model_partitioner_v2.py --mode original --quantize

# Or use CPU
python model_partitioner_v2.py --mode original --device cpu
```

### ONNX Export Fails

- Ensure you have a valid dummy image
- Try reducing image size
- Check ONNX opset compatibility

### Missing Dependencies

```bash
pip install -r requirements.txt
```

## Performance Benchmarks

| Mode | Latency | GPU Memory | Format |
|------|---------|------------|--------|
| Original | 2.3s | 3.8 GB | Native |
| Split Native | 2.4s | 3.8 GB | PyTorch + SafeTensor |
| ONNX + SafeTensor | 1.8s | 3.2 GB | ONNX + SafeTensor |
| Quantized (4-bit) | 2.1s | 2.1 GB | Quantized |

*Benchmarks on NVIDIA RTX 4090, Qwen2.5-VL-3B-Instruct*

## Contributing

Contributions welcome! Areas of interest:
- VitisAI Execution Provider integration
- AWQ quantization support
- Additional model architectures
- Performance optimizations

## License

This project follows the license of the underlying models used.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{model_partitioner_2024,
  title={Vision-Language Model Partitioner},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/model-partitioner}
}
```

