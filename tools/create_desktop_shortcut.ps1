[CmdletBinding()]
param(
    [string]$ShortcutPath = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Remit.lnk")
)

$ErrorActionPreference = "Stop"

# 桌面壳必须经 start_desktop.vbs 启动：uv 生成的 venv 里 pythonw.exe 是控制台子系统
# 的 trampoline，直接指向它会在启动瞬间分配控制台，Windows 默认终端为 Windows
# Terminal 时会弹出一个空白终端窗口。wscript.exe 是 GUI 子系统进程，配合
# WScript.Shell.Run 的隐藏窗口样式可以避免这个窗口。
$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot "start_desktop.vbs"
$Wscript = Join-Path $env:SystemRoot "System32\wscript.exe"
$Icon = Join-Path $Root "assets\remit-m-icon.ico"

foreach ($required in @($Launcher, $Wscript, $Icon)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file not found: $required"
    }
}

$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $Wscript
$shortcut.Arguments = "`"$Launcher`""
$shortcut.WorkingDirectory = $Root
$shortcut.IconLocation = "$Icon,0"
$shortcut.Description = "Remit desktop shell (developer build)"
$shortcut.WindowStyle = 1
$shortcut.Save()

Write-Host "[OK] Desktop shortcut written: $ShortcutPath"
Write-Host "[OK] Target: $Wscript `"$Launcher`""
