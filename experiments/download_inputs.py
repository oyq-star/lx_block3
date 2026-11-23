"""Download official public dumps and portable client tools, without changing host Python."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/vendor'))
import gdown
import requests

def main():
    p=argparse.ArgumentParser(); p.add_argument('kind',choices=['theia','cadets','postgres']);args=p.parse_args()
    records={
        'theia':('1p7HoH5SdMEFi0vkrEtMeG7B-hpw741p6','data/downloads/theia_e3.dump'),
        'cadets':('1DGcGBhpavNmXTnCDd_s4NWBNh2n4-6nd','data/downloads/cadets_e3.dump')}
    start=time.monotonic();last=[0]
    def progress(current,total):
        if time.monotonic()-last[0]>=20:
            print(json.dumps(dict(kind=args.kind,bytes=current,total=total,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            last[0]=time.monotonic()
    if args.kind in records:
        fid,rel=records[args.kind];dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            result=gdown.download(id=fid,output=str(dest),quiet=True,use_cookies=False,
                                  resume=True,timeout=(20,60),retries=1,progress=progress)
            if not result:raise RuntimeError('Download returned no file')
        provenance={'source':'https://github.com/ubc-provenance/PIDSMaker/blob/main/download_datasets.sh','drive_file_id':fid}
    else:
        url='https://get.enterprisedb.com/postgresql/postgresql-18.7-1-windows-x64-binaries.zip'
        dest=ROOT/'tools/downloads/postgresql-18.7-1-windows-x64-binaries.zip';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            with requests.get(url,stream=True,timeout=(20,60)) as r:
                r.raise_for_status();total=int(r.headers.get('Content-Length',0)) or None;count=0
                with dest.with_suffix('.part').open('wb') as f:
                    for chunk in r.iter_content(2**20):
                        f.write(chunk);count+=len(chunk);progress(count,total)
            dest.with_suffix('.part').replace(dest)
        provenance={'source':'https://www.enterprisedb.com/download-postgresql-binaries','download_url':url}
    with dest.open('rb') as f:header=f.read(32).hex()
    info=dict(kind=args.kind,path=str(dest),size_bytes=dest.stat().st_size,header_hex=header,
              elapsed_seconds=time.monotonic()-start,provenance=provenance)
    dest.with_suffix(dest.suffix+'.metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
