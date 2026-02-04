############# 1.) Extract dataset ############# (DONE)
python "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset_scripts\sepsis2019_psv_to_icu_long.py" `
  --psv_dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\sepsis2019_psv" `
  --pattern "*.psv" `
  --out_csv "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\icu_long.csv" `
  --out_labels_csv "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\sepsis_labels.csv" `
  --base_date "2025-01-01"


cd "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019"

python "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset_scripts\make_dataset.py" `
  --long_csv "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\icu_long.csv" `
  --out_train_jsonl "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\train.jsonl" `
  --out_val_jsonl "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\val.jsonl" `
  --preset "2019" `
  --labels_csv "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\sepsis_labels.csv"

############# 2.) CREATE VIRTUAL ENVIROMENT #############
conda create -n MedGemma python=3.10 -y
conda activate MedGemma
############# 3.) INSTALL REQUIREMENTS.TXT #############
pip install -r requirements.txt
############# 4.) Check_CUDA #############
python "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\check_cuda.py"
############# 5.) SET HUGGING FACE API KEY #############
$env:HF_TOKEN = ""
############# 6.) RUN MODEL lora #############
cd "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\model_scripts"
python train_lora.py `
  --train_jsonl "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\train.jsonl" `
  --val_jsonl   "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\val.jsonl" `
  --out_dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\results\lora\medgemma4b_lora"
  --max_len 1024 `
  --batch 1 `
  --grad_accum 8 `
  --epochs 1
############# OR Qlora #############
cd "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\model_scripts"
python train_Qlora.py `
  --train_jsonl "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\train.jsonl" `
  --val_jsonl   "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\val.jsonl" `
  --out_dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\results\Qlora\medgemma4b_Qlora"
  --max_len 1024 `
  --batch 1 `
  --grad_accum 8 `
  --epochs 1
############# 7.) EVALUATE MODEL #############
python evaluate.py \
  --base_model "google\medgemma-1.5-4b-it" \
  --val_jsonl "path\to\val.jsonl" \
  --out_dir "path\to\results" \
  --max_samples 200 \
  --max_new 320 \
  --run "lora_fp16|medgemma4b_icu_lora_fp16|lora" \
  --run "qlora_4bit|medgemma4b_icu_qlora_4bit|qlora"