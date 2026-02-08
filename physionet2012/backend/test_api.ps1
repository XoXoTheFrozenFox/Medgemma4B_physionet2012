conda run -n MedGemma --no-capture-output python -u .\medgemma_api.py `
    --base-model "google/medgemma-1.5-4b-it" `
    --adapter-dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\Qlora\checkpoint-2586" `
    --processor-dir "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\Qlora" `
    --use-4bit `
    --device cuda `
    --host 127.0.0.1 `
    --port 8000 `
    --gpu-semaphore 1

$resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/analyze" -ContentType "application/json" -Body (@{preset="quick";note="55F, fever 39.1C, cough, RR 28, SpO2 90%, WBC 16.2, CRP 120, wheeze/coarse breath sounds RLL. CXR pending.";debug=$true} | ConvertTo-Json)
$resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/analyze" -ContentType "application/json" -Body (@{preset="quick";note="55F, fever 39.1C, cough, RR 28, SpO2 90%, WBC 16.2, CRP 120, wheeze/coarse breath sounds RLL. CXR pending.";debug=$true} | ConvertTo-Json); "`n===== REPLY =====`n$($resp.reply)`n"; "`n===== META =====`n$($resp.meta | ConvertTo-Json -Depth 10)`n"
$resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/analyze" -ContentType "application/json" -Body (@{preset="normal";note="55F, fever 39.1C, cough, RR 28, SpO2 90%, WBC 16.2, CRP 120, wheeze/coarse breath sounds RLL. CXR pending.";debug=$true} | ConvertTo-Json); "`n===== REPLY =====`n$($resp.reply)`n"; "`n===== META =====`n$($resp.meta | ConvertTo-Json -Depth 10)`n"
$resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/analyze" -ContentType "application/json" -Body (@{preset="detailed";note="55F, fever 39.1C, cough, RR 28, SpO2 90%, WBC 16.2, CRP 120, wheeze/coarse breath sounds RLL. CXR pending.";debug=$true} | ConvertTo-Json); "`n===== REPLY =====`n$($resp.reply)`n"; "`n===== META =====`n$($resp.meta | ConvertTo-Json -Depth 10)`n"
