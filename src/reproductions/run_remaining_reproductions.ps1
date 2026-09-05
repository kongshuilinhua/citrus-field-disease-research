#requires -Version 5.1
<#
.SYNOPSIS
串行运行剩余复现实验，并把每次运行保存为不可覆盖的独立记录。

.DESCRIPTION
默认输出到 D:\病虫害论文\复现实验记录\<RunId>。
每次运行保留：
- 控制台标准输出与错误输出；
- 训练权重、结果 CSV、args.yaml、图片等脚本原生产物；
- 本次使用的脚本与 YAML 快照；
- Python/torch/CUDA/GPU 环境；
- 各实验开始时间、结束时间和退出码；
- 全部记录文件的 SHA-256。

如果 RunId 已存在，脚本会拒绝启动，防止覆盖旧结果。

.EXAMPLE
.\run_remaining_reproductions.ps1

.EXAMPLE
.\run_remaining_reproductions.ps1 -Experiments simclr,cms28

.EXAMPLE
.\run_remaining_reproductions.ps1 -Experiments official36
#>

[CmdletBinding()]
param(
    [ValidateSet('leafonly', 'simclr', 'cms28', 'scidb_baseline', 'rebuild36', 'dann', 'official36')]
    [string[]]$Experiments = @('leafonly', 'simclr', 'cms28', 'dann', 'scidb_baseline', 'rebuild36', 'official36'),

    [string]$RunId = (Get-Date -Format 'yyyyMMdd_HHmmss'),

    [string]$RecordBase = 'D:\病虫害论文\复现实验记录'
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = 'C:\Users\Windows11\Desktop\新版柑橘'
$reproRoot = Join-Path $workspaceRoot '复现_论文'
$pythonExe = 'D:\anaconda\envs\minidector\python.exe'
$runRoot = Join-Path $RecordBase $RunId

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python 环境不存在：$pythonExe"
}
if (Test-Path -LiteralPath $runRoot) {
    throw "运行目录已存在，为防止覆盖已停止：$runRoot"
}

New-Item -ItemType Directory -Path $runRoot | Out-Null
$logRoot = New-Item -ItemType Directory -Path (Join-Path $runRoot 'logs')
$sourceRoot = New-Item -ItemType Directory -Path (Join-Path $runRoot 'source_snapshot')

$env:CITRUS_RUN_ROOT = $runRoot
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$definitions = @{
    leafonly = @{ Script = 'leafonlysam\run_leafonlysam.py'; Description = '[53] Leaf-Only-SAM 六张样例' }
    simclr = @{ Script = 'simclr_pretrain.py'; Description = '[55] 无泄漏 SimCLR' }
    cms28 = @{ Script = 'train_28_real.py'; Description = '[28] CMS 低保真重建分类' }
    scidb_baseline = @{ Script = 'train_scidb_yolov8det.py'; Description = '[36] SciDB YOLOv8n 检测基线' }
    rebuild36 = @{ Script = 'train_36_real.py'; Description = '[36] 三通道低保真重建检测' }
    dann = @{ Script = 'dann_uda_reproduce.py'; Description = '[35] DANN 重跑并保存权重' }
    official36 = @{ Script = 'train_36_original.py'; Description = '[36] 作者补充代码，YOLOv8s 规模' }
}

$sourceFiles = @(
    'simclr_pretrain.py',
    'train_28_real.py',
    'train_scidb_yolov8det.py',
    'train_36_real.py',
    'train_36_original.py',
    'dann_uda_reproduce.py',
    'generate_experiment_screenshots.py',
    'yolov8n-cms.yaml',
    'yolov8n-3ch.yaml',
    'leafonlysam\run_leafonlysam.py',
    'threechannel_36\ultralytics-main\ultralytics\cfg\models\v8\yolov8-powerneck-c2f.yaml'
)
foreach ($relativePath in $sourceFiles) {
    $sourcePath = Join-Path $reproRoot $relativePath
    if (Test-Path -LiteralPath $sourcePath) {
        $safeName = $relativePath -replace '[\\/]', '__'
        Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $sourceRoot $safeName)
    }
}
$customSnapshot = New-Item -ItemType Directory -Path (Join-Path $sourceRoot 'custom_modules')
Get-ChildItem -LiteralPath (Join-Path $reproRoot 'custom_modules') -Filter '*.py' -File |
    Copy-Item -Destination $customSnapshot

$environmentText = & $pythonExe -c "import json,platform,sys,torch,torchvision,ultralytics,sklearn; print(json.dumps({'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'sklearn':sklearn.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},ensure_ascii=False))"
if ($LASTEXITCODE -ne 0 -or -not $environmentText) {
    throw '无法采集 Python 运行环境，实验尚未启动。'
}
$environmentText | Set-Content -LiteralPath (Join-Path $runRoot 'environment.json') -Encoding UTF8
$environmentInfo = $environmentText | ConvertFrom-Json
if (-not $environmentInfo.cuda_available) {
    throw 'CUDA 不可用，实验尚未启动。'
}
$recordDrive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($RecordBase).TrimEnd(':\')) -ErrorAction Stop
if ($recordDrive.Free -lt 25GB) {
    throw "记录盘剩余空间不足 25 GiB，实验尚未启动：$([math]::Round($recordDrive.Free / 1GB, 2)) GiB"
}
$missingScripts = @()
foreach ($experiment in $Experiments) {
    $candidate = Join-Path $reproRoot $definitions[$experiment].Script
    if (-not (Test-Path -LiteralPath $candidate)) {
        $missingScripts += $candidate
    }
}
if ($missingScripts.Count -gt 0) {
    throw "实验脚本缺失：$($missingScripts -join '；')"
}
$preflight = [ordered]@{
    checked_at = (Get-Date).ToString('o')
    cuda_available = $environmentInfo.cuda_available
    gpu = $environmentInfo.gpu
    record_drive_free_gib = [math]::Round($recordDrive.Free / 1GB, 2)
    minimum_free_gib = 25
    selected_experiments = $Experiments
    scripts_present = $true
    passed = $true
}
$preflight | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $runRoot 'preflight.json') -Encoding UTF8

$runStarted = Get-Date
$statuses = @()
$queueStatusPath = Join-Path $runRoot 'queue_status.json'

function Write-QueueStatus {
    param(
        [string]$State,
        [AllowNull()][string]$CurrentExperiment
    )
    $queueStatus = [ordered]@{
        run_id = $RunId
        state = $State
        updated_at = (Get-Date).ToString('o')
        current_experiment = $CurrentExperiment
        planned_experiments = $Experiments
        completed_experiments = @($statuses | ForEach-Object { $_.experiment })
    }
    $queueStatus | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath $queueStatusPath -Encoding UTF8
}

Write-QueueStatus -State 'running' -CurrentExperiment $null

foreach ($experiment in $Experiments) {
    $definition = $definitions[$experiment]
    $scriptPath = Join-Path $reproRoot $definition.Script
    $stdoutPath = Join-Path $logRoot "$experiment.stdout.log"
    $stderrPath = Join-Path $logRoot "$experiment.stderr.log"
    $started = Get-Date

    Write-QueueStatus -State 'running' -CurrentExperiment $experiment
    Write-Host "[$($started.ToString('yyyy-MM-dd HH:mm:ss'))] 开始：$($definition.Description)"
    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @('-u', $scriptPath) `
        -WorkingDirectory $reproRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    $ended = Get-Date

    $screenshotExitCode = $null
    $screenshotStdout = Join-Path $logRoot "$experiment.screenshots.stdout.log"
    $screenshotStderr = Join-Path $logRoot "$experiment.screenshots.stderr.log"
    try {
        $screenshotProcess = Start-Process `
            -FilePath $pythonExe `
            -ArgumentList @(
                '-u',
                (Join-Path $reproRoot 'generate_experiment_screenshots.py'),
                '--run-root',
                $runRoot,
                '--experiment',
                $experiment,
                '--exit-code',
                [string]$process.ExitCode
            ) `
            -WorkingDirectory $reproRoot `
            -RedirectStandardOutput $screenshotStdout `
            -RedirectStandardError $screenshotStderr `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        $screenshotExitCode = $screenshotProcess.ExitCode
    }
    catch {
        $_ | Out-String | Set-Content -LiteralPath $screenshotStderr -Encoding UTF8
        $screenshotExitCode = -1
    }

    $status = [ordered]@{
        experiment = $experiment
        description = $definition.Description
        script = $definition.Script
        command = "$pythonExe -u `"$scriptPath`""
        started = $started.ToString('o')
        ended = $ended.ToString('o')
        elapsed_seconds = [math]::Round(($ended - $started).TotalSeconds, 3)
        exit_code = $process.ExitCode
        screenshot_exit_code = $screenshotExitCode
        screenshots = "screenshots\$experiment"
        stdout = "logs\$experiment.stdout.log"
        stderr = "logs\$experiment.stderr.log"
    }
    $statuses += [pscustomobject]$status
    $status | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $runRoot "$experiment.status.json") -Encoding UTF8
    Write-QueueStatus -State 'running' -CurrentExperiment $null

    Write-Host "[$($ended.ToString('yyyy-MM-dd HH:mm:ss'))] 结束：$experiment，exit=$($process.ExitCode)"
}

$runEnded = Get-Date
$runMeta = [ordered]@{
    run_id = $RunId
    record_root = $runRoot
    workspace = $workspaceRoot
    started = $runStarted.ToString('o')
    ended = $runEnded.ToString('o')
    elapsed_seconds = [math]::Round(($runEnded - $runStarted).TotalSeconds, 3)
    experiments = $statuses
}
$runMeta | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $runRoot 'run_meta.json') -Encoding UTF8
Write-QueueStatus -State 'completed' -CurrentExperiment $null

$hashes = Get-ChildItem -LiteralPath $runRoot -File -Recurse |
    Where-Object { $_.Name -ne 'sha256_manifest.json' } |
    ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        [pscustomobject]@{
            path = $_.FullName.Substring($runRoot.Length + 1)
            bytes = $_.Length
            sha256 = $hash.Hash.ToLowerInvariant()
        }
    }
$hashes | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $runRoot 'sha256_manifest.json') -Encoding UTF8

Write-Host "全部记录已保存：$runRoot"
