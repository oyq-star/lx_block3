"""Bounded public-data access checks; do not log credentials or cookies."""
import json
import time
from pathlib import Path
import requests

targets={
 'theia_dump':'https://drive.google.com/uc?export=download&id=1p7HoH5SdMEFi0vkrEtMeG7B-hpw741p6',
 'theia_raw_first':'https://drive.google.com/uc?export=download&id=1yksk4I9DBnl0Tca_Y0eaiNcpRNccB9sF',
}
out=Path('data/access');out.mkdir(parents=True,exist_ok=True)
results={}
for name,url in targets.items():
    start=time.monotonic()
    try:
        with requests.get(url,stream=True,timeout=(15,25)) as r:
            chunk=next(r.iter_content(65536),b'')
            results[name]={'status':r.status_code,'content_type':r.headers.get('Content-Type'),
                           'length':r.headers.get('Content-Length'),
                           'disposition':r.headers.get('Content-Disposition'),
                           'prefix_hex':chunk[:12].hex(),'seconds':time.monotonic()-start}
            if 'text' in r.headers.get('Content-Type',''):
                (out/(name+'.html')).write_bytes(chunk)
    except Exception as e:
        results[name]={'error':str(e),'seconds':time.monotonic()-start}
    print(name,results[name],flush=True)
(out/'probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
