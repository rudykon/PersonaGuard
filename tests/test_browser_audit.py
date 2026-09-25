"""Differential tests for the browser engine against the public Python resolver.

Requires Node.js; no npm dependencies, research data, or network access.
"""
import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_protocol_replay as replay
import build_browser_demo_data as builder


class BrowserAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('node'):
            raise RuntimeError('Node.js is required for browser/Python consistency checks.')
        cls.data = json.loads(builder.render().removeprefix(builder.PREFIX).strip().removesuffix(';'))
        cls.protocol = cls.data['protocol']

    def browser(self, records, protocol=None):
        code = """
const fs=require('fs'),engine=require(process.argv[1]);
const input=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(input.records.map(record=>{
 try{return {ok:true,...engine.resolve(input.data,record)}}
 catch(error){return {ok:false,error:error.message}}
})));
"""
        data = copy.deepcopy(self.data)
        if protocol is not None:
            data['protocol'] = protocol
        result = subprocess.run(['node', '-e', code, str(ROOT / 'docs/assets/audit-engine.js')],
            input=json.dumps({'data': data, 'records': records}), text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_generated_inputs_are_current(self):
        self.assertEqual((ROOT/'docs/assets/audit-data.js').read_text(), builder.render())
        self.assertEqual(len(self.data['samples']), 11)

    def test_all_records_and_each_field_value_match_python(self):
        records = []
        vocab = {'deployment_stage': self.data['stages'], 'route_capability': self.data['capabilities']}
        vocab.update({'evidence.'+key: values for key, values in self.protocol['evidence_fields'].items()})
        for sample in self.data['samples']:
            records.append(sample['record'])
            for path, values in vocab.items():
                for value in values:
                    record = copy.deepcopy(sample['record'])
                    replay.set_field(record, path, value)
                    records.append(record)
        observed = self.browser(records)
        for record, actual in zip(records, observed):
            self.assertTrue(actual['ok'])
            self.assertEqual(actual['result'], replay.evaluate_case(record, self.protocol))
        self.assertGreater(len(records), 500)

    def test_combined_conditions_priorities_and_fallback(self):
        rng = random.Random(20260925)
        records=[]
        for _ in range(2000):
            record = copy.deepcopy(rng.choice(self.data['samples'])['record'])
            record['deployment_stage'] = rng.choice(self.data['stages'])
            record['route_capability'] = rng.choice(self.data['capabilities'])
            for field, values in self.protocol['evidence_fields'].items():
                record['evidence'][field] = rng.choice(values)
            records.append(record)
        # A positive record plus each earlier constraint verifies precedence.
        positive = copy.deepcopy(next(x['record'] for x in self.data['samples'] if x['baseline']['matched_rule_id']=='R5_BOUNDED_PASS'))
        records.append(positive)
        for path,value in [('evidence.interpretation_scope','PARTIAL'),('evidence.comparator_increment','NO_DEMONSTRATED_INCREMENT'),('evidence.user_outcome','PROXIMAL_ONLY')]:
            record=copy.deepcopy(positive);replay.set_field(record,path,value);records.append(record)
        lifecycle=copy.deepcopy(positive);lifecycle['deployment_stage']='RETENTION_TRANSFER';lifecycle['evidence']['persistence']='NOT_TESTED';lifecycle['evidence']['comparator_increment']='NO_DEMONSTRATED_INCREMENT';records.append(lifecycle)
        protocol=copy.deepcopy(self.protocol);rng.shuffle(protocol['rules'])
        observed=self.browser(records,protocol)
        actions=set()
        for record,actual in zip(records,observed):
            self.assertTrue(actual['ok'])
            self.assertEqual(actual['result'],replay.evaluate_case(record,self.protocol))
            actions.add(actual['result']['audited_action'])
            self.assertEqual([x['priority'] for x in actual['trace']], [10,20,30,40,90])
            self.assertLessEqual(sum(x['selected'] for x in actual['trace']),1)
        self.assertEqual(actions,set(self.protocol['action_labels']))

    def test_incomplete_or_invalid_inputs_never_get_permission(self):
        base=copy.deepcopy(self.data['samples'][0]['record']);records=[None,[],{},'bad']
        for field in self.protocol['required_route_fields']:
            record=copy.deepcopy(base);record.pop(field);records.append(record)
        for field in self.protocol['evidence_fields']:
            record=copy.deepcopy(base);record['evidence'][field]='INVENTED';records.append(record)
        for field in self.data['forbidden_case_fields']:
            record=copy.deepcopy(base);record[field]='PROCEED_WITHIN_EVALUATED_BOUNDARY';records.append(record)
        for key,value in [('deployment_stage','INVENTED'),('route_capability',None),('evidence',None)]:
            record=copy.deepcopy(base);record[key]=value;records.append(record)
        record=copy.deepcopy(base);record['evidence']['extra']='SUPPORTED';records.append(record)
        self.assertTrue(all(not result['ok'] for result in self.browser(records)))

    def test_original_and_hypothetical_calibration_are_distinct(self):
        record=copy.deepcopy(next(x['record'] for x in self.data['samples'] if x['record']['id']=='signed_calibration'))
        records=[copy.deepcopy(record)]
        record['evidence']['evaluation_mode']='IN_CONTEXT_USER_STUDY';record['evidence']['user_outcome']='DIRECT_TASK_AND_EXPERIENCE';records.append(copy.deepcopy(record))
        record['evidence']['interpretation_scope']='MATCHED';records.append(copy.deepcopy(record))
        self.assertEqual([x['result']['matched_rule_id'] for x in self.browser(records)],['R3_CONSEQUENCE_MATCH','R1_SCOPE_MATCH','FALLBACK'])


if __name__=='__main__': unittest.main()
