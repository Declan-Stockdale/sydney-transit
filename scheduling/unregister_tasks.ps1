<#
Stops and removes the project's scheduled tasks. Collected data is not touched.

Usage (from the project folder):
    .\scheduling\unregister_tasks.ps1
#>

foreach ($name in "SydneyTransit-CollectRealtime", "SydneyTransit-Daily", "SydneyTransit-Weather", "SydneyTransit-Radar", "SydneyTransit-Carpark") {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $name
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Removed $name"
    }
}
