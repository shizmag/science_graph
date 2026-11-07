import csv,json
import pytest
import sensitivity as s
import workflow as w

def record(q,b,score):return {'task':{'query_id':q,'evaluation_id':s.ident(q,b),'question_type':'answerable'},'result':{'faithfulness':score}}
def test_paired_missing_does_not_become_one_or_zero():
 data={p:[record('q','B3',None),record('q','B5',1),record('r','B3',.5),record('r','B5',0)] for p in ('omniroute','openrouter')}
 a=s.summarize(data,set());assert a['omniroute']['paired_n']==1;assert a['omniroute']['difference']==.5
 assert s.summarize(data,{'r'})['omniroute']['paired_n']==0

def test_requires_human_completion_and_unique_cases(tmp_path):
 p=tmp_path/'decisions.csv'; mapping=[{'case_id':'H1','query_id':'q'}]
 p.write_text('case_id,reviewer_id,adjudicator_id,category,reason,exclude_query\nH1,,,,,\n')
 with pytest.raises(ValueError,match='incomplete'):s.decisions(p,mapping)
 p.write_text('case_id,reviewer_id,adjudicator_id,category,reason,exclude_query\nH1,human1,human2,evidence_claim,conflict,yes\n')
 assert s.decisions(p,mapping)=={'q'}

def test_no_silent_packet_replacement(tmp_path):
 p=tmp_path/'x';w.write(p,'one');w.write(p,'one')
 with pytest.raises(RuntimeError):w.write(p,'two')
