param(
  [string]$BaseUrl = "http://127.0.0.1:8000",

  [ValidateSet("quick","normal","long")]
  [string]$Preset = "normal",

  [string]$Mode = "All-in-one (SOAP + Tasks + Red flags + Patient summary)",

  [int]$MaxInputLen = 1024,

  # 0 means "no cap"
  [int]$MaxTotalNewTokens = 0,

  [int]$ChunkNewTokens = 64,

  [double]$Temperature = 0.2,
  [double]$TopP = 0.95,
  [double]$RepetitionPenalty = 1.05,
  [int]$NoRepeatNgramSize = 0,

  [bool]$AutoContinue = $true,

  # <=0 disables time cap
  [double]$MaxTimeS = 0.0,

  [bool]$Concise = $true,
  [bool]$UseEndMarkerStop = $true,

  # default to NON-stream so you always see full output + meta
  [bool]$Stream = $false,

  [bool]$ShowPings = $true
)

switch ($Preset) {
  "quick" {
    $MaxTotalNewTokens = 512
    $ChunkNewTokens = 32
    $Temperature = 0.0; $TopP = 1.0; $RepetitionPenalty = 1.0
    $AutoContinue = $true; $MaxTimeS = 0.0; $Concise = $true
  }
  "normal" {
    $MaxTotalNewTokens = 1024
    $ChunkNewTokens = 64
    $Temperature = 0.2; $TopP = 0.95; $RepetitionPenalty = 1.05
    $AutoContinue = $true; $MaxTimeS = 0.0; $Concise = $true
  }
  "long" {
    $MaxTotalNewTokens = 2048
    $ChunkNewTokens = 128
    $Temperature = 0.2; $TopP = 0.95; $RepetitionPenalty = 1.1; $NoRepeatNgramSize = 3
    $AutoContinue = $true; $MaxTimeS = 0.0; $Concise = $false
  }
}

$ErrorActionPreference = "Stop"

Write-Host "== Health check ==" -ForegroundColor Cyan
$health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -TimeoutSec 10
$health | ConvertTo-Json -Depth 10

Write-Host "`n== Generate test ==" -ForegroundColor Cyan
Write-Host ("Preset={0} | max_total={1} | chunk={2} | temp={3} | max_time_s={4} | stream={5}" -f `
  $Preset, $MaxTotalNewTokens, $ChunkNewTokens, $Temperature, $MaxTimeS, $Stream) -ForegroundColor DarkCyan

$payload = @{
  note = "54yo male, fever and cough 3 days. SpO2 92% on room air, BP 98/60, HR 112, RR 28. Reports pleuritic chest discomfort."
  extra_context = "WBC 16.2, CRP 120. PMH: asthma. Allergies: NKDA. Meds: salbutamol inhaler."
  mode = $Mode

  max_input_len = $MaxInputLen
  max_total_new_tokens = $MaxTotalNewTokens

  chunk_new_tokens = $ChunkNewTokens
  auto_continue = $AutoContinue

  temperature = $Temperature
  top_p = $TopP
  repetition_penalty = $RepetitionPenalty
  no_repeat_ngram_size = $NoRepeatNgramSize

  max_time_s = $MaxTimeS
  concise = $Concise
  use_end_marker_stop = $UseEndMarkerStop
}

$json = $payload | ConvertTo-Json -Depth 6
$utf8Body = [System.Text.Encoding]::UTF8.GetBytes($json)

function Print-TextAndMeta {
  param($resp)

  Write-Host "`n--- TEXT (raw) ---" -ForegroundColor Green
  if ($null -eq $resp.text) {
    Write-Host "(null)" -ForegroundColor DarkYellow
  } else {
    Write-Host $resp.text
  }

  Write-Host "`n--- TEXT (pretty JSON if possible) ---" -ForegroundColor Green
  try {
    ($resp.text | ConvertFrom-Json) | ConvertTo-Json -Depth 20
  } catch {
    Write-Host "(not valid JSON text)" -ForegroundColor DarkYellow
  }

  Write-Host "`n--- META ---" -ForegroundColor Yellow
  $resp.meta | ConvertTo-Json -Depth 20

  if ($resp.meta.decode_debug) {
    Write-Host "`n--- DEBUG: first_32_gen_ids ---" -ForegroundColor DarkYellow
    ($resp.meta.decode_debug.first_32_gen_ids) | ConvertTo-Json -Depth 5
  }
}

function Invoke-GenerateNonStreaming {
  param([byte[]]$BodyBytes)

  $resp = Invoke-RestMethod `
    -Uri "$BaseUrl/v1/generate" `
    -Method POST `
    -Headers @{ Accept = "application/json" } `
    -ContentType "application/json; charset=utf-8" `
    -Body $BodyBytes `
    -TimeoutSec 3600

  Print-TextAndMeta -resp $resp
}

function Invoke-GenerateStreamingSSE {
  param([byte[]]$BodyBytes)

  Add-Type -AssemblyName System.Net.Http | Out-Null

  $client = New-Object System.Net.Http.HttpClient
  $client.Timeout = [TimeSpan]::FromHours(6)

  $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, "$BaseUrl/v1/generate_stream")
  $content = [System.Net.Http.ByteArrayContent]::new($BodyBytes)
  $content.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("application/json; charset=utf-8")
  $req.Content = $content

  $req.Headers.Accept.Clear()
  $req.Headers.Accept.Add([System.Net.Http.Headers.MediaTypeWithQualityHeaderValue]::new("text/event-stream"))

  $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  if (-not $resp.IsSuccessStatusCode) {
    $txt = $resp.Content.ReadAsStringAsync().Result
    throw "Streaming request failed: $($resp.StatusCode) $($resp.ReasonPhrase) :: $txt"
  }

  $stream = $resp.Content.ReadAsStreamAsync().Result
  $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

  Write-Host "`n--- LIVE TEXT ---" -ForegroundColor Green
  $full = ""
  $metaObj = $null

  try {
    while (-not $reader.EndOfStream) {
      $line = $reader.ReadLine()
      if ($null -eq $line) { continue }
      if ([string]::IsNullOrWhiteSpace($line)) { continue }

      if ($line.StartsWith(":")) {
        if ($ShowPings) { Write-Host -NoNewline "." }
        continue
      }

      if ($line.StartsWith("data:")) {
        $data = $line.Substring(5).Trim()
        if ([string]::IsNullOrWhiteSpace($data)) { continue }

        $obj = $null
        try { $obj = $data | ConvertFrom-Json } catch { continue }

        if ($obj.error) {
          Write-Host "`n[SERVER ERROR] $($obj.error)" -ForegroundColor Red
          break
        }

        if ($obj.delta) {
          $full += $obj.delta
          Write-Host -NoNewline $obj.delta
        }

        if ($obj.meta) { $metaObj = $obj.meta }
        if ($obj.done -eq $true) { break }
      }
    }
  } finally {
    Write-Host ""
    Write-Host "`n--- META ---" -ForegroundColor Yellow
    if ($metaObj) { $metaObj | ConvertTo-Json -Depth 20 }
    else { Write-Host "(no meta received)" -ForegroundColor DarkYellow }

    $reader.Dispose()
    $stream.Dispose()
    $resp.Dispose()
    $client.Dispose()
  }
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
if ($Stream) { Invoke-GenerateStreamingSSE -BodyBytes $utf8Body }
else { Invoke-GenerateNonStreaming -BodyBytes $utf8Body }
$sw.Stop()
Write-Host ("`nElapsed: {0:N2}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Magenta
