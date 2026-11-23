"""Extract only portable PostgreSQL client files; no server installation."""
from pathlib import Path
import zipfile
import subprocess

root=Path(__file__).resolve().parents[1]
archive=root/'tools/downloads/postgresql-18.7-1-windows-x64-binaries.zip'
target=root/'tools/postgresql'
with zipfile.ZipFile(archive) as z:
    names=[n for n in z.namelist() if n.startswith('pgsql/bin/') and not n.endswith('/')]
    for name in names:
        z.extract(name,target)
exe=target/'pgsql/bin/pg_restore.exe'
print('Extracted client binaries',len(names))
subprocess.run([str(exe),'--version'],check=True)
