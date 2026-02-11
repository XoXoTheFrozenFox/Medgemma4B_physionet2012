conda run -n MedGemma --no-capture-output python -u .\medgemma_api_text.py \
    --base-model "google/medgemma-1.5-4b-it" \
    --adapter-dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\Qlora\checkpoint-2586" \
    --processor-dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\Qlora" \
    --use-4bit \
    --device cuda \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-semaphore 1