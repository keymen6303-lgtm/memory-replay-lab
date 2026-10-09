import numpy as np
from scipy.linalg import expm
from src.energy import Memory,canonical,fixture,select,build_model,CONFIG


def test_exact_gram_and_unit_norms():
    x=canonical(12,50,.35)
    gram=x@x.T
    assert np.allclose(np.diag(gram),1)
    assert np.allclose((gram[:12,:12]-np.eye(12))[~np.eye(12,dtype=bool)],.35)
    assert np.array_equal(gram[12,:12],np.zeros(12))


def test_energy_gradient_and_descent():
    rng=np.random.default_rng(101);x=rng.normal(size=(8,16));x/=np.linalg.norm(x,axis=1)[:,None]
    m=Memory(x,np.arange(1,9)/3)
    z=rng.normal(size=16)/4
    grad=z-m.update(z)[0]
    fd=[]
    for i in range(16):
        step=np.zeros(16);step[i]=1e-6
        fd.append((m.energy(z+step)[0]-m.energy(z-step)[0])/2e-6)
    assert np.allclose(grad,fd,rtol=1e-6,atol=1e-8)
    states=rng.normal(size=(16,16))/4
    for _ in range(12):
        next_states=m.update(states)
        assert np.all(m.energy(next_states)<=m.energy(states)+1e-12)
        states=next_states


def test_rotation_identity():
    rng=np.random.default_rng(103);x=canonical(8,20,.35)
    w=rng.normal(size=(20,20));v=expm(.05*(w-w.T)/2)
    q=rng.normal(size=(5,20))
    assert np.allclose(Memory(x@v.T,np.ones(9)).energy(q),Memory(x,np.ones(9)).energy(q@v),atol=1e-12)
    assert np.allclose(v.T@v,np.eye(20),atol=1e-12)


def test_aggregated_mass_equals_repeated_memory():
    x=canonical(8,20,.35);q=x+.01
    m=Memory(x,np.r_[3.,np.ones(8)])
    duplicate=Memory(np.vstack((x,x[0],x[0])),np.ones(11))
    assert np.allclose(m.energy(q),duplicate.energy(q),atol=1e-12)
    assert np.allclose(m.update(q),duplicate.update(q),atol=1e-12)


def test_exact_replay_identity_and_duplicate_baseline():
    x=canonical(12,50,.35);m=Memory(x,np.ones(13));fp,_,_,_=m.recall(x)
    rng=np.random.default_rng(101);w=rng.normal(size=(50,50));v=expm(.05*(w-w.T)/2)
    shifted=Memory(x@v.T,np.ones(13));p=np.exp(8*x@fp[0]);p/=p.sum()
    delta=shifted.energy(fp[0])[0]-m.energy(fp[0])[0]
    for source in (0,1,12):
        gain=shifted.energy(fp[0])[0]-Memory(np.vstack((x@v.T,x[source])),np.ones(14)).energy(fp[0])[0]
        assert np.isclose(gain,np.log1p(p[source]*np.exp(8*delta))/8,atol=1e-12)


def test_sources_selected_without_queries_and_balanced_budget():
    f=fixture(101,24,.35)
    # Selector accepts only old energy and a precomputed permutation.
    models=[]
    for arm in ('random_replay','high_energy','low_energy'):
        model,ids,events,budget=build_model(f,arm);models.append(budget)
        assert len(ids)==len(set(ids))==8
        assert len(events)==64 and budget['total_replay_mass']==16
        assert budget['counts']==[8]*8 and budget['masses']==[2.]*8
        assert all(ids[e['slot']]==e['parent'] for e in events)
    assert len({b['persistent_array_bytes'] for b in models})==1
    assert 24 in select(f['old_energy'],f['priority'],'high_energy')
    assert 24 not in select(f['old_energy'],f['priority'],'low_energy')


def test_fixture_lures_and_read_only_recall():
    f=fixture(20268099,12,.35)
    assert len(f['query'])==9*13 and len(f['cues'])==4*13
    assert len({r['query_sha256'] for r in f['info']})==len(f['query'])
    assert not any(r['training_collision'] for r in f['info'] if r['kind'] in ('lure','unrelated'))
    for z,r in zip(f['query'],f['info']):
        if r['kind']=='lure':assert np.isclose(z@f['x'][r['parent']],np.cos(r['theta']),atol=1e-12)
    assert np.all((f['cues']!=0).sum(axis=1)==CONFIG['retained'])
    m,_,_,_=build_model(f,'high_energy');before=m.fingerprint();q=f['cues'].copy()
    m.recall(q);m.energy(f['query'])
    assert before==m.fingerprint() and np.array_equal(q,f['cues'])


def test_fixed_points_not_declared_when_iteration_fails():
    x=canonical(12,50,.35);m=Memory(x,np.ones(13))
    _,_,res,ok=m.recall(x,tol=1e-15,max_iter=1)
    assert np.all(res>1e-15) and not ok.any()
