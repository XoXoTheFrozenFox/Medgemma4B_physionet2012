############# 1.) Extract dataset ############# (DONE)
python3 "path/to/sepsis2019_psv_to_icu_long.py" `
  --psv_dir "path/to/sepsis2019_psv" `
  --pattern "*.psv" `
  --out_csv "path/to/icu_long.csv" `
  --out_labels_csv "path/to/sepsis_labels.csv" `
  --base_date "2025-01-01"

cd "/path/to/Medgemma4B_physionet2012-2019"

python3 "path/to/make_dataset.py" \
  --long_csv "dataset/icu_long.csv" \
  --out_train_jsonl "dataset/train.jsonl" \
  --out_val_jsonl "dataset/val.jsonl" \
  --preset "auto"

############# 2.) CREATE VIRTUAL ENVIROMENT #############
conda create -n MedGemma python=3.10 -y
conda activate MedGemma
############# 3.) INSTALL REQUIREMENTS.TXT #############
pip3 install -r requirements.txt
############# 4.) Check_CUDA #############
python3 "path/to/check_cuda.py"
############# 5.) SET HUGGING FACE API KEY #############
export HF_TOKEN=""
############# 6.) RUN MODEL #############
cd "path/to/model_scripts"
python3 train_lora.py `
  --train_jsonl "path/to/train.jsonl" `
  --val_jsonl   "path/to/val.jsonl" `
  --out_dir "path/to/medgemma4b_lora"
  --max_len 1024 `
  --batch 1 `
  --grad_accum 8 `
  --epochs 1
############# OR Qlora #############
cd "path/to/model_scripts"
python3 train_Qlora.py `
  --train_jsonl "path/to/train.jsonl" `
  --val_jsonl   "path/to/val.jsonl" `
  --out_dir "path/to/medgemma4b_Qlora"
  --max_len 1024 `
  --batch 1 `
  --grad_accum 8 `
  --epochs 1
############# 7.) EVALUATE MODEL #############
python3 evaluate.py \
  --base_model "google/medgemma-1.5-4b-it" \
  --val_jsonl "path/to/val.jsonl" \
  --out_dir "path/to/results" \
  --max_samples 200 \
  --max_new 320 \
  --run "lora_fp16|medgemma4b_icu_lora_fp16|lora" \
  --run "qlora_4bit|medgemma4b_icu_qlora_4bit|qlora"
