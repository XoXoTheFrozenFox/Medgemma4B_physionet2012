############# 1.) Extract dataset ############# (DONE)
python "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset_scripts\physionet2012_to_icu_long.py" `
  --set_a_dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\set-a" `
  --out_csv "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\icu_long.csv"

cd "C:\CODE ON GITHUB\Medgemma2B-4B_physionet2012-2019"
python "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset_scripts\make_dataset.py" `
  --long_csv "physionet2012\dataset\icu_long.csv" `
  --out_train_jsonl "physionet2012\dataset\train.jsonl" `
  --out_val_jsonl "physionet2012\dataset\val.jsonl"
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
cd "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\model_scripts"
python train_lora.py `
  --train_jsonl "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\train.jsonl" `
  --val_jsonl   "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\val.jsonl" `
  --out_dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\lora\medgemma4b_lora" `
  --max_len 1024 `
  --batch 1 `
  --grad_accum 8 `
  --epochs 1
############# OR Qlora #############
cd "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\model_scripts"
python train_Qlora.py `
  --train_jsonl "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\train.jsonl" `
  --val_jsonl   "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\val.jsonl" `
  --out_dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\Qlora\medgemma4b_Qlora" `
  --max_len 1024 `
  --batch 1 `
  --grad_accum 8 `
  --epochs 1
############# 7.) EVALUATE MODEL #############
python evaluate.py `
  --base_model "google/medgemma-1.5-4b-it" `
  --val_jsonl "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\val.jsonl" `
  --out_dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\eval_reports_qlora" `
  --max_samples 100 `
  --max_new 320 `
  --run "qlora_main|C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\Qlora|qlora"
