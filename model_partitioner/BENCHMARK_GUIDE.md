# Benchmark Guide

Comprehensive benchmarking suite for measuring and comparing performance across all inference modes.

## Quick Start

### Basic Benchmark (5 runs per mode)
```bash
python benchmark.py --image demo.jpg --runs 5
```

### Benchmark Specific Modes
```bash
python benchmark.py --image demo.jpg --runs 10 --modes original run_onnx
```

### Compare CUDA vs CPU
```bash
python benchmark.py --image demo.jpg --compare-devices --runs 5
```

### Compare Quantized vs Non-Quantized
```bash
python benchmark.py --image demo.jpg --compare-quantization --runs 5
```

## Metrics Measured

### 🚀 Performance Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Duration** | Total inference time | seconds |
| **Tokens/sec** | Token generation throughput | tokens/second |
| **Tokens Generated** | Number of output tokens | count |
| **Vision Inference Time** | Vision encoder latency (ONNX mode) | seconds |

### 💾 Memory Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **CPU Memory Delta** | Change in CPU memory usage | MB |
| **GPU Memory Delta** | Change in GPU memory usage | MB |
| **GPU Peak Memory** | Peak GPU memory during inference | MB |

### 📦 Model Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Vision Model Size** | Size of vision model on disk | MB |
| **Language Model Size** | Size of language model on disk | MB |
| **ONNX Model Size** | Size of ONNX vision model | MB |

### 📊 Statistical Metrics

For each metric, the following statistics are calculated:
- **Mean**: Average across all runs
- **Std**: Standard deviation
- **Min**: Minimum value
- **Max**: Maximum value
- **Median**: Median value

## Usage Examples

### 1. Standard Benchmark

Benchmark all default modes with 5 runs each:

```bash
python benchmark.py \
    --image demo.jpg \
    --text "Describe this image" \
    --runs 5
```

**Output:**
```
======================================================================
BENCHMARK RESULTS SUMMARY
======================================================================

Mode            Metric                    Mean         Std          Min          Max         
--------------------------------------------------------------------------------------
original        Duration (s)              2.345        0.123        2.201        2.489       
original        Tokens/sec                54.67        2.13         52.34        56.78       
original        Tokens Generated          128.000      0.000        128.000      128.000     
original        GPU Peak Mem (MB)         3845.234     12.456       3820.123     3860.567    

split_native    Duration (s)              2.412        0.134        2.267        2.534       
split_native    Tokens/sec                53.09        1.98         51.23        55.12       
...
```

### 2. High-Precision Benchmark

Run more iterations for better statistics:

```bash
python benchmark.py \
    --image demo.jpg \
    --runs 20 \
    --max-tokens 256
```

### 3. Quick Benchmark (No Warmup)

Skip warmup run for faster results:

```bash
python benchmark.py \
    --image demo.jpg \
    --runs 3 \
    --no-warmup
```

### 4. Single Mode Benchmark

Benchmark only one specific mode:

```bash
python benchmark.py \
    --image demo.jpg \
    --modes run_onnx \
    --runs 10
```

### 5. Save Results

Save benchmark results to specific files:

```bash
# JSON format
python benchmark.py \
    --image demo.jpg \
    --runs 5 \
    --output my_benchmark_results.json

# JSON + CSV format
python benchmark.py \
    --image demo.jpg \
    --runs 5 \
    --output my_benchmark_results.json \
    --csv
```

### 6. Device Comparison

Compare performance between CUDA and CPU:

```bash
python benchmark.py \
    --image demo.jpg \
    --compare-devices \
    --runs 5
```

**Output:**
- `benchmark_cuda_TIMESTAMP.json`
- `benchmark_cpu_TIMESTAMP.json`

### 7. Quantization Comparison

Compare quantized vs non-quantized models:

```bash
python benchmark.py \
    --image demo.jpg \
    --compare-quantization \
    --runs 5 \
    --device cuda
```

**Output:**
- `benchmark_non_quantized_TIMESTAMP.json`
- `benchmark_quantized_TIMESTAMP.json`

### 8. Custom Model

Benchmark a different model:

```bash
python benchmark.py \
    --image demo.jpg \
    --model-id "Qwen/Qwen2-VL-7B-Instruct" \
    --runs 5
```

## Command-Line Options

### Required Options

```bash
--image PATH              # Input image for testing
```

### Mode Selection

```bash
--modes [MODE ...]        # Modes to benchmark
                          # Choices: original, split_native, run_onnx
                          # Default: all three modes

# Examples:
--modes original          # Only original mode
--modes original run_onnx # Two modes
```

### Benchmark Configuration

```bash
--runs N                  # Number of runs per mode (default: 5)
--max-tokens N            # Max tokens to generate (default: 128)
--no-warmup              # Skip warmup run
```

### Model Configuration

```bash
--model-id ID            # HuggingFace model ID
--device {auto|cuda|cpu} # Device to use (default: auto)
--quantize               # Enable 4-bit quantization
--output-dir PATH        # Output directory (default: split_models)
```

### Output Options

```bash
--output FILE            # Save results to JSON file
--csv                    # Also save as CSV
```

### Comparison Modes

```bash
--compare-devices        # Compare CUDA vs CPU
--compare-quantization   # Compare quantized vs non-quantized
```

## Understanding the Output

### Console Output

#### 1. Results Table

```
Mode            Metric                    Mean         Std          Min          Max         
--------------------------------------------------------------------------------------
original        Duration (s)              2.345        0.123        2.201        2.489       
original        Tokens/sec                54.67        2.13         52.34        56.78       
```

- **Mean**: Average performance across all runs
- **Std**: Variation between runs (lower is more consistent)
- **Min**: Best case performance
- **Max**: Worst case performance

#### 2. Performance Comparison

```
⚡ Fastest Mode: run_onnx (1.845s)

Relative Performance (vs run_onnx):
  run_onnx       : 1.845s (1.00x, +0.0%)
  original       : 2.345s (0.79x, +27.1%)
  split_native   : 2.412s (0.76x, +30.7%)
```

- Shows which mode is fastest
- Relative speedup compared to fastest
- Percentage difference from baseline

#### 3. Throughput Comparison

```
run_onnx       : 69.35 tokens/sec
original       : 54.67 tokens/sec
split_native   : 53.09 tokens/sec
```

- Token generation throughput
- Higher is better

### JSON Output

```json
{
  "original": {
    "duration": {
      "mean": 2.345,
      "std": 0.123,
      "min": 2.201,
      "max": 2.489,
      "median": 2.340,
      "values": [2.201, 2.345, 2.340, 2.489, 2.350]
    },
    "tokens_per_sec": {
      "mean": 54.67,
      ...
    },
    "num_runs": 5
  },
  ...
}
```

### CSV Output

```csv
Mode,Metric,Mean,Std,Min,Max,Median,Num Runs
original,duration,2.3450,0.1230,2.2010,2.4890,2.3400,5
original,tokens_per_sec,54.6700,2.1300,52.3400,56.7800,54.5600,5
```

## Interpreting Results

### Performance Analysis

#### 1. Latency (Duration)

- **Lower is better**
- Look at **mean** for typical performance
- Check **std** for consistency
- Compare **min** across modes for best case

**Good performance:**
```
Duration: 1.5s ± 0.05s  (mean ± std)
```

**Inconsistent performance:**
```
Duration: 2.3s ± 0.5s   (high variation)
```

#### 2. Throughput (Tokens/sec)

- **Higher is better**
- Indicates generation speed
- Important for production use

**Good throughput:**
```
Tokens/sec: 100+ tokens/sec on GPU
Tokens/sec: 20+ tokens/sec on CPU
```

#### 3. Memory Usage

- **Lower is better** (especially on GPU)
- GPU Peak shows maximum VRAM needed
- CPU Delta shows system memory impact

**Typical values (3B model):**
```
GPU Peak: 3-4 GB (non-quantized)
GPU Peak: 2-3 GB (quantized)
CPU Memory: 500-1000 MB
```

### Comparison Guidelines

#### Original vs ONNX

**Expected results:**
- ONNX should be **10-30% faster** for vision processing
- Similar or slightly better overall latency
- Comparable memory usage

#### Quantized vs Non-Quantized

**Expected results:**
- **40-50% less GPU memory** (quantized)
- **10-20% slower** inference (quantized)
- Good trade-off for memory-constrained systems

#### CUDA vs CPU

**Expected results:**
- **5-10x faster** on CUDA
- **10-20x more memory** on CUDA
- CPU suitable for development/testing

## Best Practices

### 1. Run Multiple Iterations

```bash
# At least 5 runs for reliability
python benchmark.py --runs 5

# 10+ runs for high precision
python benchmark.py --runs 10
```

### 2. Use Warmup

```bash
# Default includes warmup (recommended)
python benchmark.py --runs 5

# Only skip for quick tests
python benchmark.py --runs 3 --no-warmup
```

### 3. Consistent Test Conditions

- Use same image across tests
- Close other GPU applications
- Run on idle system
- Use same prompt length

### 4. Save Results

```bash
# Always save for later comparison
python benchmark.py --runs 5 --output results.json --csv
```

### 5. Compare Systematically

```bash
# First: Baseline
python benchmark.py --modes original --runs 10 --output baseline.json

# Then: Optimized versions
python benchmark.py --modes run_onnx --runs 10 --output onnx.json

# Compare JSON files
```

## Troubleshooting

### High Variance (Std)

**Problem:** Standard deviation > 10% of mean

**Solutions:**
- Increase number of runs
- Close background applications
- Check system temperature/throttling
- Ensure GPU isn't being used by others

### Unexpectedly Slow

**Problem:** Performance worse than expected

**Solutions:**
- Check device (CUDA vs CPU)
- Verify model is loaded correctly
- Check for quantization overhead
- Monitor system resources

### Memory Issues

**Problem:** CUDA out of memory

**Solutions:**
```bash
# Use quantization
python benchmark.py --quantize

# Use CPU
python benchmark.py --device cpu

# Reduce max tokens
python benchmark.py --max-tokens 64
```

### Inconsistent Results

**Problem:** Results vary significantly between runs

**Solutions:**
- Increase warmup iterations
- Check thermal throttling
- Reduce system load
- Use longer benchmark runs

## Example Workflow

### Complete Performance Analysis

```bash
# 1. Baseline benchmark
python benchmark.py \
    --image demo.jpg \
    --runs 10 \
    --output baseline.json \
    --csv

# 2. Compare devices
python benchmark.py \
    --image demo.jpg \
    --compare-devices \
    --runs 10

# 3. Compare quantization
python benchmark.py \
    --image demo.jpg \
    --compare-quantization \
    --runs 10 \
    --device cuda

# 4. Test different modes
python benchmark.py \
    --image demo.jpg \
    --modes original \
    --runs 20 \
    --output original_detailed.json

python benchmark.py \
    --image demo.jpg \
    --modes run_onnx \
    --runs 20 \
    --output onnx_detailed.json
```

## Performance Targets

### Development (CPU)
- Duration: < 10s
- Tokens/sec: > 10
- CPU Memory: < 2GB

### Production (GPU)
- Duration: < 2s
- Tokens/sec: > 50
- GPU Memory: < 4GB

### Optimized (GPU + ONNX)
- Duration: < 1.5s
- Tokens/sec: > 70
- GPU Memory: < 3GB

## Advanced Usage

### Custom Metrics

Modify `benchmark.py` to add custom metrics:

```python
# Add in benchmark_original_mode()
metrics['custom_metric'] = calculate_custom_metric()
```

### Batch Benchmarking

```bash
# Benchmark multiple images
for img in images/*.jpg; do
    python benchmark.py --image "$img" --runs 5 --output "results_$(basename $img).json"
done
```

### Automated Comparison

```python
# Compare JSON results
import json

with open('baseline.json') as f:
    baseline = json.load(f)

with open('optimized.json') as f:
    optimized = json.load(f)

speedup = baseline['original']['duration']['mean'] / optimized['run_onnx']['duration']['mean']
print(f"Speedup: {speedup:.2f}x")
```

## Reporting Results

### Include in Reports

1. **Test Configuration**
   - Model ID
   - Device (GPU model or CPU)
   - Image resolution
   - Prompt used

2. **Key Metrics**
   - Duration (mean ± std)
   - Throughput (tokens/sec)
   - Memory usage

3. **Comparison**
   - Relative speedup
   - Memory savings
   - Trade-offs

### Example Report Format

```
Benchmark Results: Qwen2.5-VL-3B-Instruct
Hardware: NVIDIA RTX 4090, CUDA 12.1
Image: 1920x1080, demo.jpg
Runs: 10 iterations

Mode          Duration      Throughput    GPU Memory
-------------------------------------------------
Original      2.34s ± 0.12  54.7 tok/s   3.85 GB
ONNX          1.85s ± 0.09  69.4 tok/s   3.62 GB
Quantized     2.12s ± 0.15  60.4 tok/s   2.13 GB

Conclusions:
- ONNX provides 21% latency improvement
- Quantization reduces GPU memory by 45%
- Recommended: ONNX for production deployment
```

Happy benchmarking! 🚀

