conda run -n MedGemma python \
  "/home/heystekgrobler/Documents/Medgemma4B_physionet2012-2019/physionet2012/backend/medgemma_api.py" \
  --base-model "google/medgemma-1.5-4b-it" `
  --adapter-dir "/home/heystekgrobler/Documents/Medgemma4B_physionet2012-2019/physionet2012/results/Qlora/checkpoint-2586" \
  --processor-dir "/home/heystekgrobler/Documents/Medgemma4B_physionet2012-2019/physionet2012/results/Qlora" \
  --use-4bit \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-semaphore 1 \
  --warmup \
  --allow-origins "*"

conda run -n MedGemma --no-capture-output python -u \
  "/mnt/c/CODE ON GITHUB/Medgemma4B_physionet2012-2019/physionet2012/backend/medgemma_api.py" \
  --base-model "google/medgemma-1.5-4b-it" \
  --adapter-dir "/mnt/c/CODE ON GITHUB/Medgemma4B_physionet2012-2019/physionet2012/results/Qlora/checkpoint-2586" \
  --processor-dir "/mnt/c/CODE ON GITHUB/Medgemma4B_physionet2012-2019/physionet2012/results/Qlora" \
  --use-4bit \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-semaphore 1 \
  --warmup \
  --allow-origins "*"