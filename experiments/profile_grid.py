"""High-resolution cached inference measurements on representative frozen models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch

from train_eval import ROOT, EventGRU, cache_arrays, score_cache


def main():
    run=ROOT/'runs/full_grid_v1'
    device=torch.device('cuda')
    torch.set_num_threads(2)
    results=[]
    with TemporaryDirectory(dir=str(run),prefix='profile_') as temp:
        for dataset in ('theia','cadets'):
            for mode in ('clean','block'):
                checkpoint=torch.load(run/dataset/('17_'+mode)/'best.pt',map_location=device)
                config=checkpoint['config']
                model=EventGRU(config['dataset']['vocabulary_size'],config['embedding'],config['hidden']).to(device)
                model.load_state_dict(checkpoint['model'])
                for view in ('clean','global_relation'):
                    folder=ROOT/'data/cache'/(dataset+'_longoutage_v1')/('test_'+view)
                    arrays=cache_arrays(folder)
                    torch.cuda.reset_peak_memory_stats()
                    _,stats=score_cache(model,arrays,device,1024,Path(temp)/(dataset+'_'+mode+'_'+view+'.npy'))
                    row=dict(dataset=dataset,mode=mode,seed=17,view=view,batch_size=1024,**stats,
                             torch_cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                    results.append(row)
                    print(json.dumps(row),flush=True)
    payload=dict(device=torch.cuda.get_device_name(0),clock=str(time.get_clock_info('perf_counter')),
                 scope='cache reads, CUDA forward and CPU score aggregation; excludes raw ingest/context construction',
                 rows=results)
    (run/'inference_profile.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


if __name__=='__main__': main()
