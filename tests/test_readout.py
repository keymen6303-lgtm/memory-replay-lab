import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from src.energy import Memory,canonical,fixture,build_model
from src.readout import readouts,decomposition,calibrate,cached_case,fingerprint,READOUTS


def test_mass_normalization_is_constant_and_keeps_decisions():
    f=fixture(101,24,.35);m,_,_,_=build_model(f,'high_energy')
    q=f['query'];normal=Memory(m.vectors,m.masses/m.masses.sum(),m.beta)
    s=-m.energy(q);sn=-normal.energy(q);shift=np.log(m.masses.sum())/m.beta
    assert np.allclose(sn,s-shift,rtol=0,atol=1e-12)
    assert np.allclose(normal.update(q),m.update(q),rtol=0,atol=1e-12)
    t=np.quantile(s[:25],.1);tn=np.quantile(sn[:25],.1)
    assert np.array_equal(s>=t,sn>=tn)
    labels=np.arange(len(s))<25
    assert roc_auc_score(labels,s)==roc_auc_score(labels,sn)


def test_decomposition_distinguishes_support_and_mass():
    f=fixture(103,24,.35);m,_,_,_=build_model(f,'high_energy')
    base,support,weight,total=decomposition(m,f['query'],25)
    assert np.allclose(total,support+weight,atol=1e-12)
    assert np.allclose(base+total,-m.energy(f['query']),atol=1e-12)
    assert (support>0).all() and (weight>0).all()
    equal=Memory(m.vectors,np.ones(len(m.vectors)),m.beta)
    assert np.array_equal(decomposition(equal,f['query'],25)[2],np.zeros(len(base)))


def test_readouts_do_not_mutate_model_or_recall_dynamics():
    f=fixture(101,24,.35);m,_,_,_=build_model(f,'random_replay')
    before=m.fingerprint();query=f['query'].copy();cues=f['cues'].copy()
    partial0=m.recall(cues)[0]
    scores,*_=readouts(m,query)
    partial1=m.recall(cues)[0]
    assert scores.shape==(4,225)
    assert np.array_equal(partial0,partial1)
    assert m.fingerprint()==before and np.array_equal(query,f['query'])


def test_reconstruction_uses_only_presented_query_and_stored_model():
    x=canonical(8,20,.35);m=Memory(x,np.ones(9));q=x[[0,8]]+.02
    scores,full,*_=readouts(m,q)
    assert np.array_equal(scores[2],-np.mean((full-q)**2,axis=1))
    assert np.array_equal(scores[3],(q@m.vectors.T).max(axis=1))
    assert np.isfinite(scores).all()


def test_thresholds_use_only_calibration_targets():
    f=fixture(101,24,.35);arrays={}
    for arm in ('no_replay','random_replay','high_energy','low_energy'):
        m,_,_,_=build_model(f,arm);arrays[arm+'_scores']=readouts(m,f['query'])[0]
    t=calibrate([(arrays,{})])
    for arm in t:
        for ri,name in enumerate(READOUTS):
            assert t[arm][name]==np.quantile(arrays[arm+'_scores'][ri,:25],.1)
        arrays[arm+'_scores'][:,25:]=1e12
    assert t==calibrate([(arrays,{})])


def test_cache_detects_source_and_raw_array_changes(tmp_path,monkeypatch):
    import src.readout as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    def tiny(seed):
        return {'toy':np.array([seed])},{'seed':seed}
    monkeypatch.setattr(module,'evaluate_case',tiny)
    fp={'sha256':'fixture-code'}
    a,_,hit=cached_case(101,'dev',fp)
    assert not hit
    b,_,hit=cached_case(101,'dev',fp,resume=True)
    assert hit and np.array_equal(a['toy'],b['toy'])
    with pytest.raises(ValueError):cached_case(101,'dev',{'sha256':'changed'},resume=True)
    raw=tmp_path/'work/readout/cases/dev_101.npz';raw.write_bytes(b'bad')
    with pytest.raises(ValueError):cached_case(101,'dev',fp,resume=True)
