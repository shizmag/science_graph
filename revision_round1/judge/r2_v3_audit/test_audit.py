import importlib.util, pathlib
p=pathlib.Path(__file__).with_name('audit.py');spec=importlib.util.spec_from_file_location('audit_v3',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def test_agreement():
 assert a.icc([0,.5,1],[0,.5,1])==1
 assert a.kappa([0,.5,1],[0,.5,1])==1
 assert a.icc([1,1],[1,1]) is None
 assert a.kappa([1,1],[1,1]) is None
 assert a.spearman([1,1],[1,1]) is None

def test_cluster_bootstrap_retains_groups():
 x=a.agreement([('a',0,0),('a',1,1),('b',0,1)],'faithfulness',100)
 assert x['n']==3 and x['queries']==2 and abs(x['mae']-1/3)<1e-12
 assert 0<=x['cluster_ci95']['mae'][0]<=x['cluster_ci95']['mae'][1]<=1
