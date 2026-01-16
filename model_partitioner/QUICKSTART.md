# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Download a test image (or use your own)
# Place it as demo.jpg in the current directory
```

## Basic Workflow

### Step 1: Test Original Model (Baseline)

```bash
python model_partitioner_v2.py \
    --mode original \
    --image demo.jpg \
    --text "What is in this image?"
```

**Expected output:**
- Model loads successfully
- Generates description of the image
- Shows performance metrics

### Step 2: Split the Model

```bash
python model_partitioner_v2.py \
    --mode split_native \
    --image demo.jpg \
    --text "What is in this image?"
```

**What happens:**
- Model is split into vision + language components
- Files saved to `split_models/` directory
- Runs inference with split models
- Compares performance

### Step 3: Convert Vision to ONNX

```bash
python model_partitioner_v2.py \
    --mode convert_onnx \
    --image demo.jpg
```

**What happens:**
- Vision model exported to ONNX format
- Model validated and tested
- File saved to `split_models/onnx_model/vision_model.onnx`

### Step 4: Run with ONNX

```bash
python model_partitioner_v2.py \
    --mode run_onnx \
    --image demo.jpg \
    --text "What is in this image?"
```

**What happens:**
- Uses ONNX vision model
- Uses SafeTensor language model
- Shows performance improvements

### Step 5: Create Standalone Scripts

```bash
python model_partitioner_v2.py \
    --mode save_standalone \
    --image demo.jpg
```

**What happens:**
- Creates independent inference scripts
- Vision and language can be tested separately
- Useful for debugging

## One-Command Run All

```bash
python model_partitioner_v2.py \
    --mode all \
    --image demo.jpg \
    --text "Describe this image in detail."
```

This runs all steps sequentially and compares results!

## Common Options

### Use GPU if available
```bash
python model_partitioner_v2.py --mode original --device cuda --image demo.jpg
```

### Force CPU
```bash
python model_partitioner_v2.py --mode original --device cpu --image demo.jpg
```

### Use Quantization (save memory)
```bash
python model_partitioner_v2.py --mode original --quantize --image demo.jpg
```

### Generate longer responses
```bash
python model_partitioner_v2.py \
    --mode original \
    --image demo.jpg \
    --text "Describe every detail you see" \
    --max-tokens 256
```

### Use different model
```bash
python model_partitioner_v2.py \
    --mode original \
    --model-id "Qwen/Qwen2-VL-7B-Instruct" \
    --image demo.jpg
```

## Output Structure

After running, you'll have:

```
split_models/
├── vision_model/
│   └── vision_model.pt              # Vision weights (PyTorch)
├── language_model/
│   └── language_model.safetensors   # Language weights (SafeTensor)
├── onnx_model/
│   └── vision_model.onnx            # Vision ONNX model
├── standalone/
│   ├── vision/
│   │   ├── vision_model.onnx
│   │   └── vision_inference.py
│   └── language/
│       ├── language_model.safetensors
│       └── language_inference.py
└── model_config.json                # Configuration metadata
```

## Testing Standalone Scripts

### Test Vision Model Only

```bash
cd split_models/standalone/vision
python vision_inference.py --image ../../../demo.jpg
```

### Test Language Model Only

```bash
cd split_models/standalone/language
python language_inference.py --text "Explain quantum computing"
```

## Performance Comparison

Run this to compare all modes:

```bash
# Original
python model_partitioner_v2.py --mode original --image demo.jpg > results_original.txt

# Split Native
python model_partitioner_v2.py --mode split_native --image demo.jpg > results_split.txt

# ONNX
python model_partitioner_v2.py --mode run_onnx --image demo.jpg > results_onnx.txt

# Compare the performance metrics in each file
```

## Next Steps

1. **Integrate VitisAI**: Replace ONNX CPU execution with VitisAI EP
2. **Add AWQ**: Use AWQ quantization for language model
3. **Optimize**: Fine-tune performance for your hardware
4. **Deploy**: Use standalone scripts in production

## Troubleshooting

### "demo.jpg not found"
- Download or create a test image
- Or specify your image: `--image path/to/your/image.jpg`

### "CUDA out of memory"
- Use `--quantize` flag
- Or use `--device cpu`
- Or use a smaller model

### "Module not found"
- Run: `pip install -r requirements.txt`

### ONNX conversion fails
- Ensure you have onnx and onnxruntime installed
- Check that the image path is correct

## Getting Help

- Check `README.md` for detailed documentation
- See `requirements.txt` for dependencies
- Review error messages for specific issues

## Example Session

```bash
# Complete workflow example
$ python model_partitioner_v2.py --mode all --image demo.jpg --text "What do you see?"

======================================================================
MODEL PARTITIONER V2
======================================================================
Mode: all
Model: Qwen/Qwen2.5-VL-3B-Instruct
Device: auto
======================================================================

# ... runs all modes ...

✅ COMPLETE
```

Happy partitioning! 🚀

