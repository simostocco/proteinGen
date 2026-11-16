"""Read Windows power event metadata; preserve recovery evidence on D:."""
import datetime
import json
from pathlib import Path
import subprocess

root = Path('/mnt/d/Simone/proteinGen_data/e017_v95_expanded_validation_20261028_v1/recovery_20261029_v1')
command = 'Get-WinEvent -FilterHashtable @{LogName="System";StartTime=[datetime]"2026-10-28 22:00:00";Id=41,42,1074,6005,6006,6008,1,12,13} -MaxEvents 40 -ErrorAction Stop | ForEach-Object { $_.ToXml() }'
result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], capture_output=True)
assert result.returncode == 0, result.stderr.decode(errors='replace')
with (root / 'WINDOWS_POWER_EVENTS.xml.txt').open('xb') as f:
    f.write(result.stdout)
with (root / 'TERMINATION_HOST_EVIDENCE_ADDENDUM.json').open('x') as f:
    json.dump(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evidence='WINDOWS_POWER_EVENTS.xml.txt', returncode=result.returncode,
        observation='Kernel-Power 41 and EventLog 6008 establish an unclean host shutdown/restart; WSL also reports unclean shutdown recovery.',
        limitation='Reported previous shutdown time does not align cleanly with last batch timestamp. Exact evaluator termination trigger remains unresolved; power loss, crash, forced shutdown, and separate WSL termination are not distinguished.',
        original_termination_cause='E_UNRESOLVED_EXACT_CAUSE_EXTERNAL_INTERRUPTION_SUPPORTED',
        TEST_content_read=False), f, indent=2)
print('WINDOWS_POWER_EVIDENCE_PRESERVED')
