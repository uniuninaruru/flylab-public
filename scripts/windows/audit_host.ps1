# FlyLab Windows host audit (spec section 2.3). READ-ONLY: installs nothing, changes nothing.
#
# Run on the Windows GPU host, or from the Mac over SSH without copying the file:
#   ssh <user>@<gpu-host> "powershell -NoProfile -ExecutionPolicy Bypass -Command -" < scripts/windows/audit_host.ps1
# (-ExecutionPolicy Bypass applies to that one process only; no setting is changed.)
#
# Redacts computer name, user name, profile path, IPv4 (except 0.0.0.0/127.*), IPv6 and MAC
# addresses. Tailscale peers are never listed. Review the output before sharing it.

$ErrorActionPreference = 'Continue'
$env:WSL_UTF8 = '1'   # make wsl.exe print UTF-8 instead of UTF-16
# Decode native output as UTF-8 and write UTF-8 to SSH (Japanese Windows defaults to CP932).
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$lines = New-Object System.Collections.Generic.List[string]
function Add-Line([string]$s) { $lines.Add($s) }
function Section([string]$t) { Add-Line ''; Add-Line "## $t" }
function Run([string]$label, [scriptblock]$cmd) {
    Add-Line "> $label"
    try { (& $cmd 2>&1 | Out-String -Width 200).TrimEnd() -split "`r?`n" | ForEach-Object { Add-Line $_ } }
    catch { Add-Line "(error: $($_.Exception.Message))" }
}
function Have([string]$name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

Add-Line '# FlyLab Windows host audit (read-only)'
Add-Line ("generated_utc=" + (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))
Add-Line 'script=scripts/windows/audit_host.ps1'
Add-Line 'redacted=computername,user,profile,ipv4(except 0.0.0.0/127.*),ipv6,mac'

Section 'Windows'
Run 'OS' { Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture | Format-List }
Run 'CPU' { (Get-CimInstance Win32_Processor | Select-Object -First 1).Name }
Run 'RAM (GB total / free)' {
    $os = Get-CimInstance Win32_OperatingSystem
    '{0:N1} / {1:N1}' -f ($os.TotalVisibleMemorySize / 1MB), ($os.FreePhysicalMemory / 1MB)
}
Run 'Fixed disks (GB free / total)' {
    Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
        ForEach-Object { '{0} {1:N0} / {2:N0}' -f $_.DeviceID, ($_.FreeSpace / 1GB), ($_.Size / 1GB) }
}

Section 'NVIDIA GPU (Windows driver; WSL2 and Docker reuse it)'
if (Have 'nvidia-smi') {
    Run 'nvidia-smi query' { nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv }
    $ver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($ver -match '^(\d+)\.') {
        $major = [int]$Matches[1]
        if ($major -ge 580) { Add-Line "driver_check=OK driver=$ver (>= 580 needed by torch 2.15.0 / CUDA 13.2)" }
        else { Add-Line "driver_check=TOO_OLD driver=$ver (FlyLab image needs >= 580)" }
    } else { Add-Line 'driver_check=UNKNOWN' }
} else { Add-Line 'nvidia-smi: not found' }

Section 'WSL2'
if (Have 'wsl') {
    Run 'wsl --version' { wsl.exe --version }
    Run 'wsl --status' { wsl.exe --status }
    Run 'wsl -l -v' { wsl.exe -l -v }
} else { Add-Line 'wsl: not found' }

Section 'Docker'
if (Have 'docker') {
    Run 'docker version' { docker version --format 'client={{.Client.Version}} server={{.Server.Version}} os={{.Server.Os}}' }
    Run 'docker info' { docker info --format 'os={{.OperatingSystem}} runtimes={{range $k, $v := .Runtimes}}{{$k}} {{end}}' }
    Run 'docker context ls' { docker context ls }
} else { Add-Line 'docker: not on PATH for this session (Docker Desktop may be stopped or not installed)' }

Section 'OpenSSH server and firewall exposure'
Run 'sshd service' { Get-Service sshd -ErrorAction SilentlyContinue | Select-Object Status, StartType | Format-List }
Run 'sshd_config ListenAddress / PasswordAuthentication' {
    $cfg = Join-Path $env:ProgramData 'ssh\sshd_config'
    if (Test-Path $cfg) { Select-String -Path $cfg -Pattern '^\s*(ListenAddress|PasswordAuthentication|PubkeyAuthentication|DefaultShell)\b' | ForEach-Object { $_.Line } }
    else { 'sshd_config not found' }
}
Run 'Firewall rules for port 22 (RemoteAddress Any = reachable from every network)' {
    Get-NetFirewallPortFilter -Protocol TCP -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq '22' } |
        ForEach-Object {
            $r = $_ | Get-NetFirewallRule
            $a = $r | Get-NetFirewallAddressFilter
            '{0} enabled={1} profile={2} remote={3}' -f $r.DisplayName, $r.Enabled, $r.Profile, ($a.RemoteAddress -join ',')
        }
}
Run 'Listening on TCP 22' { Get-NetTCPConnection -State Listen -LocalPort 22 -ErrorAction SilentlyContinue | Select-Object LocalAddress, LocalPort | Format-Table -AutoSize }

Section 'Tailscale (no peer list is recorded)'
$ts = Get-Command tailscale -ErrorAction SilentlyContinue
if (-not $ts) { $p = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'; if (Test-Path $p) { $ts = $p } }
if ($ts) {
    Run 'tailscale version' { & $ts version }
    Run 'backend state' { ((& $ts status --json) | ConvertFrom-Json).BackendState }
} else { Add-Line 'tailscale: not found' }

# ---- redaction ----
# Same rules, in the same order, as scripts/redact.py (tests/audit/test_audit_redaction.py runs
# both on the same cases). Over-redaction is accepted; under-redaction is a bug.
$generic = @('root', 'ubuntu', 'runner', 'user', 'admin', 'administrator', 'docker')
$secrets = @($env:COMPUTERNAME, $env:USERNAME, $env:USERPROFILE) |
    Where-Object { $_ -and ($generic -notcontains $_.ToLower()) } | Sort-Object Length -Descending
$text = $lines -join "`n"
foreach ($s in $secrets) {
    $pat = [regex]::Escape($s)
    # Names shorter than 3 characters (e.g. "yu", "pc") are redacted as whole words only.
    if ($s.Length -lt 3) { $pat = '(?<![A-Za-z0-9])' + $pat + '(?![A-Za-z0-9])' }
    $text = [regex]::Replace($text, $pat, '<redacted>', 'IgnoreCase')
}
$text = [regex]::Replace($text, '[\w.+-]+@[\w-]+\.[\w.-]+', '<email>')
$text = [regex]::Replace($text, '\b[\w-]+(?:\.[\w-]+)*\.ts\.net\b', '<tailnet-host>', 'IgnoreCase')
$text = [regex]::Replace($text, '\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b', '<mac>')
# 0.0.0.0, 127.x, netmasks (255.x) and 100.64.0.0 (Tailscale's published range) identify nobody.
$keep = { param($a) $a -eq '0.0.0.0' -or $a -eq '100.64.0.0' -or $a.StartsWith('127.') -or $a.StartsWith('255.') }
# IPv4-mapped IPv6 as printed by ss: [::ffff:203.0.113.9]. Before the IPv6 pass.
$text = [regex]::Replace($text, '(?i)(::ffff:)(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])', {
    param($m) if (& $keep $m.Groups[2].Value) { $m.Value } else { $m.Groups[1].Value + '<ipv4>' }
})
# fd7a:115c:a1e0:: is Tailscale's published IPv6 prefix, not a host address: keep it.
$text = [regex]::Replace($text, '(?<![0-9A-Za-z:])(?:(?:[0-9A-Fa-f]{1,4}:){4,7}[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*)?|::(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*))(?![0-9A-Za-z])', {
    param($m) if ($m.Value -eq 'fd7a:115c:a1e0::') { $m.Value } else { '<ipv6>' }
})
# IPv4, line by line. A number right after the word "version" (or its Japanese spelling,
# written here as regex escapes because this file must stay ASCII) is protected; four numbers
# followed by "-microsoft" are a WSL kernel version; a sentence-final period hides nothing.
$nul = [string][char]0
$text = ($text -split "`n" | ForEach-Object {
    $kept = New-Object System.Collections.Generic.List[string]
    $line = [regex]::Replace($_, '(?i)(\bversion|\u30D0\u30FC\u30B8\u30E7\u30F3)(\s*[:=]?\s*v?)(\d+(?:\.\d+)+)', {
        param($m) $kept.Add($m.Groups[3].Value); $m.Groups[1].Value + $m.Groups[2].Value + $nul + ($kept.Count - 1) + $nul
    })
    # macOS netstat style address.port
    $line = [regex]::Replace($line, '(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})\.(\d{1,5})(?![\d.])', {
        param($m) if (& $keep $m.Groups[1].Value) { $m.Value } else { '<ipv4>.' + $m.Groups[2].Value }
    })
    $line = [regex]::Replace($line, '(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?!\d)(?!\.\d)(?!-microsoft)', {
        param($m) if (& $keep $m.Groups[1].Value) { $m.Value } else { '<ipv4>' }
    })
    [regex]::Replace($line, $nul + '(\d+)' + $nul, { param($m) $kept[[int]$m.Groups[1].Value] })
}) -join "`n"
Write-Output $text
