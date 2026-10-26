<#
Registers the five Windows Task Scheduler tasks that collect data for this project.
Safe to run again: existing tasks with the same names are replaced.

    SydneyTransit-CollectRealtime
        Runs collect_realtime.py continuously. Starts at logon, and a check every
        15 minutes restarts it if it has stopped (if it is already running, the
        check does nothing, so there is never more than one copy).

    SydneyTransit-Daily
        At 3:30am every day: pull_static.py (new timetable versions), then
        compact_raw.py (bundle yesterday's snapshots). If the PC was off or asleep
        at 3:30am, it runs as soon as possible afterwards.

    SydneyTransit-Weather
        Every 6 hours: pull_weather.py (BOM station observations + Open-Meteo
        hourly weather at every rail station).

    SydneyTransit-Radar
        Every 30 minutes: pull_radar.py (BOM rain radar frames, kept only 2 hours).

    SydneyTransit-Carpark
        Every 15 minutes: pull_carpark.py (Park&Ride car park occupancy).

All run as you, only while you are logged on, with no console window
(pythonw.exe). Output goes to data\logs\. No admin rights or password needed.

Usage (from the project folder, in PowerShell):
    .\scheduling\register_tasks.ps1

Check status:   Get-ScheduledTask -TaskName "SydneyTransit-*" | Get-ScheduledTaskInfo
Run now:        Start-ScheduledTask -TaskName "SydneyTransit-Daily"
Stop collector: Stop-ScheduledTask -TaskName "SydneyTransit-CollectRealtime"
Remove both:    .\scheduling\unregister_tasks.ps1
#>

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $project ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "Virtual environment not found at $pythonw" }
$user = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

# --- Realtime collector -------------------------------------------------------
$collectAction = New-ScheduledTaskAction -Execute $pythonw -Argument "collect_realtime.py" -WorkingDirectory $project
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
$watchdog = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
$collectSettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "SydneyTransit-CollectRealtime" -TaskPath "\" -Force `
    -Description "Polls TfNSW realtime feeds (Sydney Trains + Metro) into data\raw\realtime" `
    -Action $collectAction -Trigger @($atLogon, $watchdog) -Settings $collectSettings -Principal $principal | Out-Null

# --- Daily static timetable + compaction --------------------------------------
$staticAction = New-ScheduledTaskAction -Execute $pythonw -Argument "pull_static.py" -WorkingDirectory $project
$compactAction = New-ScheduledTaskAction -Execute $pythonw -Argument "compact_raw.py" -WorkingDirectory $project
$daily = New-ScheduledTaskTrigger -Daily -At "3:30am"
$dailySettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "SydneyTransit-Daily" -TaskPath "\" -Force `
    -Description "Downloads new static GTFS versions, then compacts finished days of realtime snapshots" `
    -Action @($staticAction, $compactAction) -Trigger $daily -Settings $dailySettings -Principal $principal | Out-Null

# --- Weather: BOM stations + Open-Meteo every 6 hours, radar every 30 minutes --
function Register-Repeating($name, $script, $minutes, $description) {
    $action = New-ScheduledTaskAction -Execute $pythonw -Argument $script -WorkingDirectory $project
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
        -RepetitionInterval (New-TimeSpan -Minutes $minutes) -RepetitionDuration (New-TimeSpan -Days 3650)
    $settings = New-ScheduledTaskSettingsSet `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
        -MultipleInstances IgnoreNew `
        -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $name -TaskPath "\" -Force -Description $description `
        -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
}
Register-Repeating "SydneyTransit-Weather" "pull_weather.py" 360 "BOM station observations and Open-Meteo hourly weather at every rail station"
Register-Repeating "SydneyTransit-Radar" "pull_radar.py" 30 "BOM Sydney rain radar frames (kept only 2 hours by the BOM)"
Register-Repeating "SydneyTransit-Carpark" "pull_carpark.py" 15 "TfNSW Park&Ride car park occupancy"

Get-ScheduledTask -TaskName "SydneyTransit-*" | Format-Table TaskName, State -AutoSize
Write-Host "Registered. The collector starts at next logon, or now with:"
Write-Host '    Start-ScheduledTask -TaskName "SydneyTransit-CollectRealtime"'
