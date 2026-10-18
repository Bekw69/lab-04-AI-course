"""Language/condition aggregation and the paired prompt experiment."""

import unittest

from lab04_analysis import analyze
from lab04_checkpoint import score_record


class AnalysisTests(unittest.TestCase):
    def test_language_split_and_paired_letter_improvement(self):
        docs=[{'id':'d_'+lang,'lang':lang} for lang in ('kk','ru','en')]
        truth={d['id']:{'date':'2026-10-01','seller_name':'Қала','total':20} for d in docs}
        results,ocr,letters={},{},{}
        import json
        for d in docs:
            for cond in ('clean','degraded'):
                key=d['id'],cond
                pred=dict(truth[d['id']])
                if d['lang']=='kk':
                    pred['seller_name']='Кала'
                results[key]=score_record(*key,json.dumps(pred),truth[d['id']])
                ocr[key]={'found':{field:True for field in truth[d['id']]}}
                if d['lang']=='kk':
                    letters[key]=score_record(*key,json.dumps(truth[d['id']]),truth[d['id']])
        report=analyze(docs,truth,{m:ocr for m in (3,4,6,11)},results,letters)
        self.assertEqual(report['accuracy']['clean']['all']['norm']['correct'],8)
        self.assertEqual(report['accuracy']['clean']['kk']['norm']['correct'],2)
        self.assertEqual(report['accuracy']['clean']['ru']['norm']['correct'],3)
        self.assertEqual(report['letter_experiment']['clean']['paired_names']['improved'],1)
        self.assertEqual(report['confusions']['all']['kazakh_specific'][0]['count'],2)
        self.assertEqual(report['confusions']['all']['kazakh_specific'][0]['true'],'Қ')


if __name__=='__main__':
    unittest.main()
