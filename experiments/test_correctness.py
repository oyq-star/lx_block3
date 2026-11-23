"""Meaningful invariant tests. Synthetic fixtures are not efficacy evidence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from causal_data import EventDay,draw_contexts
from evaluation import process_metrics

class Correctness(unittest.TestCase):
    def test_invisible_positive_never_earns_ap_or_budget_credit(self):
        m=process_metrics(range(6),[0,1,2],{0:.9,3:.8,1:.7,4:.7})
        self.assertAlmostEqual(m['AP_all'],.5);self.assertAlmostEqual(m['AP_observed'],.75)
        self.assertEqual(m['top_50']['alerts'],4);self.assertEqual(m['top_50']['tp'],2)
        self.assertAlmostEqual(m['top_50']['recall'],2/3)
        self.assertEqual(m['positive_unobservable'],1)

    def test_zero_positive_precedence_and_all_invisible(self):
        self.assertIsNone(process_metrics([0,1],[],{})['AP_all'])
        m=process_metrics([0,1],[0],{});self.assertEqual(m['AP_all'],0.)
        self.assertEqual(m['top_50']['alerts'],0);self.assertIsNone(m['top_50']['precision'])

    def test_actual_timestamp_causality_and_effective_mask_matching(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'day_2000-01-01';folder.mkdir()
            n=302;t=np.arange(n,dtype=np.int64)*10**9;t[-1]=t[-2]
            values=dict(t=t,actor=np.zeros(n,dtype=np.int32),object=np.ones(n,dtype=np.int32),
                        rel=(np.arange(n)%3).astype(np.int16),otype=np.full(n,2,dtype=np.uint8),seq=np.arange(n,dtype=np.int64))
            for key,a in values.items():np.save(folder/(key+'.npy'),a)
            np.save(folder/'starts.npy',np.array([0]));np.save(folder/'ends.npy',np.array([n]))
            np.save(folder/'targets.npy',np.arange(n))
            for r in range(3):np.save(folder/('rel_'+str(r)+'.npy'),np.flatnonzero(values['rel']==r))
            (folder/'INDEX_COMPLETE').write_text('complete')
            day=EventDay(temp,'2000-01-01',context=32,lookback_seconds=180)
            clean=day.clean_positions(n-1)
            self.assertTrue(np.all(t[clean]<t[-1]));self.assertNotIn(n-2,clean)
            exercised=0
            for seed in range(100):
                cl,b,summary=draw_contexts(day,n-1,'block',seed,0,(60,),0)
                for mode in ['iid','dispersed']:
                    clean2,aug,s2=draw_contexts(day,n-1,mode,seed,0,(60,),0)
                    self.assertTrue(np.array_equal(cl,clean2));self.assertEqual(summary['K_raw'],s2['K_raw'])
                    self.assertEqual(len(np.setdiff1d(cl,aug)),summary['K_eff'])
                    self.assertTrue(np.all(t[aug]<t[-1]));self.assertLessEqual(len(aug),32)
                    if mode=='dispersed' and summary['K_eff']:
                        removed=np.setdiff1d(cl,aug)
                        self.assertTrue(np.all(values['rel'][removed]==summary['relation']))
                exercised+=int(summary['K_eff']>0)
            self.assertGreater(exercised,5)
            day.close()

if __name__=='__main__':unittest.main()
