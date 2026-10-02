#!/usr/bin/bash

# This file is useful to download models from HuggingFace and save the weights on
# the default cache.  To do so, run the following command in the terminal
# from the root directory of the project.
# ```bash
# $ bash scripts/models/download_models.sh
# ```
# Note that some models are gated and require asking access and authentifying
# to HuggingFace beforehand. Check the models documentations for more details.

MODELS=(
    "google/gemma-3-1b-pt"
    "google/gemma-3-4b-pt"
    "google/gemma-3-12b-pt"
    "google/gemma-3-27b-pt"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-8B-Base"
    "apple/aimv2-large-patch14-224"
    "apple/aimv2-huge-patch14-224"
    "apple/aimv2-1B-patch14-224"
    "apple/aimv2-3B-patch14-224"
    "facebook/dinov2-small"
    "facebook/dinov2-base"
    "facebook/dinov2-large"
    "facebook/dinov2-giant"
    "facebook/dinov2-with-registers-small"
    "facebook/dinov2-with-registers-base"
    "facebook/dinov2-with-registers-large"
    "facebook/dinov2-with-registers-giant"
    "facebook/dinov3-vits16-pretrain-lvd1689m"
    "facebook/dinov3-vitb16-pretrain-lvd1689m"
    "facebook/dinov3-vitl16-pretrain-lvd1689m"
    "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    "google/vit-base-patch16-224-in21k"
    "google/vit-large-patch16-224-in21k"
    "google/vit-huge-patch14-224-in21k"
)
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Download HuggingFace Models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for model in "${MODELS[@]}"; do

    printf "📥 %s\n" "$model" >/dev/tty
    
    # Local path for logs
    TMP_LOG="./hf_download_temp.log"

    # Capture both stdout and stderr to check for errors
    # if output=$(hf download "$model" 2>&1); then
    if hf download "$model" 2>&1 | tee "$TMP_LOG"; then
        echo " ✅ SUCCESS"
        ((SUCCESS_COUNT++))
    else
        echo " ❌ FAILED"
        ((FAILED_COUNT++))
        FAILED_MODELS+=("$model")
        
        # Determine and display the likely cause
        if [[ "$output" == *"401"* ]] || [[ "$output" == *"Unauthorized"* ]]; then
            echo "   ⚠️  Unauthorized: Model may require authentication"
            echo "   → Run 'hf auth login' and verify access on Hugging Face"
        elif [[ "$output" == *"404"* ]] || [[ "$output" == *"Not Found"* ]]; then
            echo "   ⚠️  Model not found: Verify the repository name is correct"
            echo "   → Check: https://huggingface.co/$model"
        elif [[ "$output" == *"403"* ]] || [[ "$output" == *"Forbidden"* ]]; then
            echo "   ⚠️  Access denied: Model is gated or requires permission"
            echo "   → Visit https://huggingface.co/$model and request access"
        elif [[ "$output" == *"timeout"* ]] || [[ "$output" == *"Connection"* ]]; then
            echo "   ⚠️  Network error: Check your proxy/VPN connection"
            echo "   → Verify network settings and try again"
        else
            echo "   ⚠️  Download failed: $output"
        fi
    fi

    # Delete log file  
    rm -f "$TMP_LOG"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Successful: $SUCCESS_COUNT"
echo "  ❌ Failed:     $FAILED_COUNT"
echo ""

if [ $FAILED_COUNT -gt 0 ]; then
    echo "  Failed models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "    - $model"
    done
    echo ""
    echo "  Troubleshooting:"
    echo "    1. Verify model name at https://huggingface.co"
    echo "    2. Run 'hf auth login' for gated models"
    echo "    3. Request access on each model's Hugging Face page"
    echo "    4. Check your network/proxy configuration"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📁 Models cached in: ~/.cache/huggingface/hub/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Exit with non-zero code if any failures (for CI/CD pipelines)
[ $FAILED_COUNT -eq 0 ]
