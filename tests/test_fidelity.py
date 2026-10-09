import numpy as np
import pytest
from src.energy import fixture,build_model,Memory
from src.fidelity import source_vectors,construct,recognize,cached_case,calibrate,key,GROUPS


def test_independent_sources_match_angles_and_have_separate_random_stream():
    f=fixture(101,24,.35);sources=source_vectors(f,101)
    assert np.allclose(np.linalg.norm(sources['independent'],axis=1),1,rtol=0,atol=1e-12)
    assert np.allclose(np.sum(f['x']*sources['independent'],axis=1),np.sum(f['x']*f['y'],axis=1),rtol=0,atol=1e-12)
    assert not np.array_equal(sources['independent'],f['y'])
    assert np.array_equal(source_vectors(f,101)['independent'],sources['independent'])
    f['query']=np.full_like(f['query'],100.)
    assert np.array_equal(source_vectors(f,101)['independent'],sources['independent'])


def test_exact_reuses_original_and_sources_share_budget_and_selection():
    f=fixture(103,24,.35);sources=source_vectors(f,103)
    for arm in ('random_replay','high_energy','low_energy'):
        original,ids,events,_=build_model(f,arm)
        exact,chosen,ev=construct(f,'exact',arm,sources)
        assert np.array_equal(exact.vectors,original.vectors)
        assert np.array_equal(exact.masses,original.masses)
        assert np.array_equal(chosen,ids) and ev==events
        for source in ('coherent','independent'):
            m,chosen,ev=construct(f,source,arm,sources)
            assert np.array_equal(chosen,ids) and ev==events
            assert np.array_equal(m.masses,original.masses)
            assert m.vectors.nbytes==original.vectors.nbytes
        assert len(np.unique(exact.vectors,axis=0))==33
        assert len(np.unique(construct(f,'coherent',arm,sources)[0].vectors,axis=0))==25


def test_coherent_duplicates_have_known_max_and_weighted_identities():
    f=fixture(101,24,.35);sources=source_vectors(f,101)
    base=build_model(f,'no_replay')[0]
    for arm in ('random_replay','high_energy','low_energy'):
        model,ids,_=construct(f,'coherent',arm,sources)
        assert np.allclose(recognize(model,f['query'])[1],recognize(base,f['query'])[1],rtol=0,atol=1e-12)
        masses=np.ones(25);masses[ids]+=2
        collapsed=Memory(f['y'],masses)
        assert np.allclose(model.energy(f['query']),collapsed.energy(f['query']),rtol=0,atol=1e-12)
        assert np.allclose(model.update(f['cues']),collapsed.update(f['cues']),rtol=0,atol=1e-12)


def test_recognition_receives_only_stored_model_and_query_is_read_only():
    f=fixture(101,24,.35);model=build_model(f,'random_replay')[0]
    before=model.fingerprint();q=f['query'].copy()
    scores=recognize(model,q)
    assert np.array_equal(scores[0],-model.energy(q))
    assert np.array_equal(scores[1],(q@model.vectors.T).max(axis=1))
    assert model.fingerprint()==before and np.array_equal(q,f['query'])


def test_fidelity_thresholds_ignore_non_targets():
    arrays={key(s,a)+'_scores':np.arange(450).reshape(2,225)+j for j,(s,a) in enumerate(GROUPS)}
    t=calibrate([(arrays,{})])
    for s,a in GROUPS:
        assert t[key(s,a)]['weighted']==np.quantile(arrays[key(s,a)+'_scores'][0,:25],.1)
        arrays[key(s,a)+'_scores'][:,25:]=10**10
    assert t==calibrate([(arrays,{})])


def test_fidelity_cache_refuses_changed_source_and_corrupted_npz(tmp_path,monkeypatch):
    import src.fidelity as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'evaluate_case',lambda seed:({'value':np.array([seed])},{'seed':seed}))
    fp={'sha256':'locked'}
    a,_,hit=cached_case(101,'dev',fp);assert not hit
    b,_,hit=cached_case(101,'dev',fp,True);assert hit and np.array_equal(a['value'],b['value'])
    with pytest.raises(ValueError):cached_case(101,'dev',{'sha256':'changed'},True)
    (tmp_path/'work/fidelity/cases/dev_101.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):cached_case(101,'dev',fp,True)
