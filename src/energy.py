"""Modern Hopfield closed-form replication and prospectively specified extension.

This model stores explicit vectors and masses. It is neither STDP nor a trained
MHN Boltzmann machine; adding mass represents replay in the measure of Eq. 13.
"""
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'outputs/energy'
ARMS = ('no_replay', 'random_replay', 'high_energy', 'low_energy')
LABELS = dict(zip(ARMS, ('无回放', '随机回放', '高能量优先', '低能量优先')))
CAL_SEEDS = list(range(20269001, 20269021))
CONF_SEEDS = list(range(20270001, 20270041))
CONDITIONS = [(n, c) for n in (12, 24, 40) for c in (.15, .35, .65)]
CONFIG = dict(d=64, beta=8., epsilon=.05, slots=8, events=64,
              gain=.25, angles=[.12, .24, .48], directions=2,
              views=4, retained=16, tol=1e-10, max_iter=500,
              replication_seed=20268001, replication_rotations=2000,
              primary_n=24, primary_c=.35, primary_angle=.24)


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def condition_id(n, c):
    return f'n{n}_c{round(c*100):02d}'


def canonical(n, d, c):
    if not (0 < c < 1 and n + 2 <= d):
        raise ValueError('invalid cluster geometry')
    x = np.zeros((n + 1, d))
    x[:n, 0] = np.sqrt(c)
    x[np.arange(n), np.arange(1, n + 1)] = np.sqrt(1 - c)
    x[n, n + 1] = 1.
    return x


@dataclass
class Memory:
    vectors: np.ndarray
    masses: np.ndarray
    beta: float = 8.

    def __post_init__(self):
        if self.vectors.ndim != 2 or self.masses.shape != (len(self.vectors),):
            raise ValueError('inconsistent memory arrays')
        if not np.all(self.masses > 0):
            raise ValueError('masses must be positive')

    def energy(self, query):
        q = np.atleast_2d(query)
        l = self.beta * (q @ self.vectors.T) + np.log(self.masses)
        return .5 * (q*q).sum(axis=1) - logsumexp(l, axis=1) / self.beta

    def update(self, query):
        q = np.atleast_2d(query)
        return softmax(self.beta * (q @ self.vectors.T) + np.log(self.masses),
                       axis=1) @ self.vectors

    def recall(self, query, tol=1e-10, max_iter=500):
        q = np.atleast_2d(query).copy()
        done = np.zeros(len(q), bool); steps = np.zeros(len(q), int)
        for _ in range(max_iter):
            active = np.flatnonzero(~done)
            if not len(active): break
            nxt = self.update(q[active])
            distance = np.linalg.norm(nxt-q[active], axis=1)
            q[active] = nxt; steps[active] += 1
            done[active[distance <= tol]] = True
        residual = np.linalg.norm(self.update(q)-q, axis=1)
        return q, steps, residual, residual <= tol

    def fingerprint(self):
        h = hashlib.sha256()
        for x in (self.vectors, self.masses): h.update(x.tobytes())
        h.update(np.float64(self.beta).tobytes())
        return h.hexdigest()


def bootstrap(values, n=10000, seed=2026101007, level=.95):
    v = np.asarray(values, float)
    if not len(v) or not np.all(np.isfinite(v)): raise ValueError('invalid CI data')
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    a = (1-level)/2
    return np.quantile(means, [a, 1-a]).tolist()


def replicate(destination=DEST):
    out = Path(destination); (out/'tables').mkdir(parents=True, exist_ok=True)
    x = canonical(12, 50, .35); m = Memory(x, np.ones(13))
    fp, _, res, done = m.recall(x, tol=1e-13, max_iter=1000)
    if not done.all(): raise RuntimeError('replication fixed points failed')
    e = m.energy(fp); p = softmax(8*fp@x.T, axis=1)
    baseline = np.log1p(p)/8
    # target axis first, replay-source axis second
    diagonal = np.arange(12)
    off = ~np.eye(12, dtype=bool)
    def families(g):
        return np.array([g[12, 12], g[diagonal, diagonal].mean(),
                         g[:12, :12][off].mean(), g[:12, 12].mean(),
                         g[12, :12].mean()])
    rng = np.random.default_rng(CONFIG['replication_seed'])
    rows = []; max_identity_error = 0.; orth_error = 0.
    started = time.perf_counter()
    for rotation in range(CONFIG['replication_rotations']):
        w = rng.normal(size=(50,50)); omega = (w-w.T)/2
        for epsilon in (.02, .05):
            v = expm(epsilon*omega); y = x@v.T
            orth_error = max(orth_error, float(np.abs(v.T@v-np.eye(50)).max()))
            e2 = Memory(y, np.ones(13)).energy(fp)
            delta = e2-e
            # Direct Eq.13 computation, all 13 replay sources.
            raw = np.empty((13,13))
            for r in range(13):
                raw[:,r] = e2-Memory(np.vstack((y,x[r])),np.ones(14)).energy(fp)
            exact = np.logaddexp(0, np.log(p)+8*delta[:,None])/8
            max_identity_error = max(max_identity_error, float(np.abs(raw-exact).max()))
            g = families(raw-baseline)
            rows.append(dict(rotation=rotation, epsilon=epsilon,
                outlier_rise=delta[12], cluster_rise=delta[:12].mean(),
                outlier_self=g[0], cluster_self=g[1], within_cluster=g[2],
                outlier_to_cluster=g[3], cluster_to_outlier=g[4]))
        if (rotation+1)%500==0: print(f'原文闭式复现：{rotation+1}/2000旋转',flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(out/'tables/replication_rotations.csv',index=False)
    summaries = []
    for eps,g in frame.groupby('epsilon'):
        diffs = dict(outlier_minus_cluster_rise=g.outlier_rise-g.cluster_rise,
            outlier_self_minus_cluster_self=g.outlier_self-g.cluster_self,
            cluster_self_minus_within=g.cluster_self-g.within_cluster,
            within_minus_outlier_to_cluster=g.within_cluster-g.outlier_to_cluster,
            within_minus_cluster_to_outlier=g.within_cluster-g.cluster_to_outlier)
        summaries.append(dict(epsilon=eps, n_rotations=len(g),
            means=g.drop(columns=['rotation','epsilon']).mean().to_dict(),
            paired={k:dict(mean=float(z.mean()),ci95=bootstrap(z,n=5000,
                    seed=2026101001)) for k,z in diffs.items()}))
    stability = []
    for f,prob in zip(fp,p):
        jac = 8*(x.T@(prob[:,None]*x)-np.outer(f,f))
        stability.append(float(np.linalg.eigvalsh(jac).max()))
    checks = dict(fixed_point_max_residual=float(res.max()),
                  max_jacobian_eigenvalue=max(stability),
                  exact_replay_identity_max_error=max_identity_error,
                  orthogonality_max_error=orth_error,
                  all_ordering_ci_lower_positive=all(
                    z['ci95'][0]>0 for s in summaries for z in s['paired'].values()))
    answer = dict(source='https://arxiv.org/html/2605.27975v2',
        scope='原文闭式等角簇实验方向复现；未复现MHN-BM或扩散模型',
        original_energy=m.energy(x).tolist(), fixed_point_energy=e.tolist(),
        summaries=summaries, checks=checks, seconds=time.perf_counter()-started)
    dump(out/'tables/replication_summary.json',answer)
    np.savez_compressed(out/'tables/replication_arrays.npz',x=x,fp=fp,p=p)
    return answer


def fixture(seed, n, c):
    streams = [np.random.default_rng(s) for s in
               np.random.SeedSequence([seed,n,round(c*1000)]).spawn(6)]
    orient, rotation, directions, masks, choosing, events = streams
    d=CONFIG['d']; x=canonical(n,d,c)
    q,r=np.linalg.qr(orient.normal(size=(d,d)))
    q=q*np.where(np.diag(r)>=0,1.,-1.)
    x=x@q.T
    w=rotation.normal(size=(d,d)); v=expm(CONFIG['epsilon']*(w-w.T)/2)
    y=x@v.T
    original=Memory(x,np.ones(len(x)),CONFIG['beta'])
    old_energy=original.energy(x)
    priority=choosing.permutation(len(x))
    # Each slot has exactly 8 events, with the slot schedule common across arms.
    schedule=np.concatenate([events.permutation(CONFIG['slots']) for _ in range(8)])
    query=[]; info=[]
    def add(z,kind,parent,theta=0.,direction=-1):
        query.append(z); info.append(dict(kind=kind,parent=parent,theta=theta,
                                         direction=direction,query_id=len(query)-1))
    for i,z in enumerate(x): add(z,'target',i)
    for i,z in enumerate(x):
        for k in range(CONFIG['directions']):
            u=directions.normal(size=d); u-=u@z*z; u/=np.linalg.norm(u)
            for angle in CONFIG['angles']:
                add(np.cos(angle)*z+np.sin(angle)*u,'lure',i,angle,k)
    for i in range(len(x)):
        z=directions.normal(size=d); z/=np.linalg.norm(z); add(z,'unrelated',-1)
    for i,z in enumerate(y): add(z,'task2',i)
    query=np.array(query)
    distances=2-2*(query@np.vstack((x,y)).T)
    for row,dist in zip(info,distances):
        row['min_training_distance']=float(np.sqrt(max(0.,dist.min())))
        row['training_collision']=bool(dist.min()<1e-12)
        row['query_sha256']=hashlib.sha256(query[row['query_id']].tobytes()).hexdigest()
    cues=[]; cue_info=[]
    for i,z in enumerate(x):
        for k in range(CONFIG['views']):
            kept=masks.choice(d,CONFIG['retained'],replace=False)
            cue=np.zeros(d);cue[kept]=z[kept];cues.append(cue)
            cue_info.append(dict(parent=i,view=k,kept=json.dumps(sorted(kept.tolist()))))
    return dict(x=x,y=y,v=v,old_energy=old_energy,priority=priority,
                schedule=schedule,query=query,info=info,cues=np.array(cues),cue_info=cue_info)


def select(old_energy, priority, arm, slots=8):
    rank=np.empty(len(priority),int);rank[priority]=np.arange(len(priority))
    e=np.round(old_energy,12)
    if arm=='random_replay': order=priority
    elif arm=='high_energy': order=np.lexsort((rank,-e))
    elif arm=='low_energy': order=np.lexsort((rank,e))
    elif arm=='no_replay': return np.empty(0,int)
    else: raise ValueError('unknown arm')
    return order[:slots].copy()


def build_model(f,arm):
    ids=select(f['old_energy'],f['priority'],arm,CONFIG['slots'])
    masses=np.zeros(len(ids));counts=np.zeros(len(ids),dtype=np.int64);events=[]
    if len(ids):
        for ordinal,slot in enumerate(f['schedule']):
            masses[slot]+=CONFIG['gain']; counts[slot]+=1
            events.append(dict(event=ordinal,slot=slot,parent=int(ids[slot]),
                gain=CONFIG['gain'],slot_count=int(counts[slot]),mass_after=float(masses[slot])))
        vectors=np.vstack((f['y'],f['x'][ids]));weights=np.r_[np.ones(len(f['y'])),masses]
    else: vectors=f['y'].copy();weights=np.ones(len(vectors))
    model=Memory(vectors,weights,CONFIG['beta'])
    # Includes explicit vectors, implicit base masses materialized in this
    # implementation, buffer ID/count arrays, beta/gain scalars. Logs excluded.
    byte_count=vectors.nbytes+weights.nbytes+ids.nbytes+counts.nbytes+16
    return model,ids,events,dict(persistent_array_bytes=byte_count,
        total_replay_mass=float(masses.sum()),event_count=len(events),
        buffer_count=len(ids),counts=counts.tolist(),masses=masses.tolist())


def evaluate(seed,n,c,cohort,thresholds=None,save_arrays=False):
    start=time.perf_counter(); f=fixture(seed,n,c); base=dict(seed=seed,n_cluster=n,
        correlation=c,condition=condition_id(n,c),cohort=cohort)
    arrays={k:f[k] for k in ('x','y','v','old_energy','priority','schedule','query','cues')}
    complete=[];partial=[];events=[];resources=[]
    for arm in ARMS:
        build_start=time.perf_counter(); model,ids,ev,budget=build_model(f,arm)
        selection_seconds=time.perf_counter()-build_start
        before=model.fingerprint();scores=-model.energy(f['query'])
        recovered,steps,residual,converged=model.recall(f['cues'],
            tol=CONFIG['tol'],max_iter=CONFIG['max_iter'])
        nearest=(recovered@f['x'].T).argmax(axis=1)
        for i,(meta,score) in enumerate(zip(f['info'],scores)):
            row={**base,'arm':arm,**meta,'score':float(score)}
            if thresholds is not None: row['judged_old']=bool(score>=thresholds[arm])
            complete.append(row)
        for i,meta in enumerate(f['cue_info']):
            parent=meta['parent']
            partial.append({**base,'arm':arm,**meta,'recalled_id':int(nearest[i]),
                'correct_identity':bool(nearest[i]==parent and converged[i]),
                'mse':float(np.mean((recovered[i]-f['x'][parent])**2)),
                'converged':bool(converged[i]),'iterations':int(steps[i]),
                'fixed_point_residual':float(residual[i])})
        events.extend({**base,'arm':arm,**e} for e in ev)
        resources.append({**base,'arm':arm,**budget,'buffer_ids':json.dumps(ids.tolist()),
            'query_read_only':before==model.fingerprint(),
            'selection_and_build_seconds':selection_seconds})
        if save_arrays:
            arrays[f'{arm}_vectors']=model.vectors;arrays[f'{arm}_masses']=model.masses
            arrays[f'{arm}_ids']=ids;arrays[f'{arm}_scores']=scores
            arrays[f'{arm}_recovered']=recovered
    elapsed=time.perf_counter()-start
    for r in resources:r['shared_total_case_seconds']=elapsed
    if save_arrays:
        path=DEST/'tables'/f'{cohort}_{condition_id(n,c)}_{seed}.npz'
        np.savez_compressed(path,**arrays)
    return dict(queries=complete,partial=partial,events=events,resources=resources)


def fingerprint():
    files=['src/energy.py','energy_protocol.md']
    return dict(files={p:sha(ROOT/p) for p in files},configuration=CONFIG,
                calibration=CAL_SEEDS,confirmation=CONF_SEEDS,
                conditions=[list(x) for x in CONDITIONS])


def cache_case(seed,n,c,cohort,fp,thresholds,resume):
    path=ROOT/'work/energy/cases'/f'{cohort}_{condition_id(n,c)}_{seed}.json'
    if resume and path.exists():
        cached=json.loads(path.read_text())
        if cached['fingerprint']!=fp or cached['thresholds']!=thresholds:
            raise ValueError(f'stale cache: {path}')
        return cached['result'],True
    save=seed==(CAL_SEEDS[0] if cohort=='calibration' else CONF_SEEDS[0]) and (n,c)==(24,.35)
    result=evaluate(seed,n,c,cohort,thresholds,save_arrays=save)
    dump(path,dict(fingerprint=fp,thresholds=thresholds,result=result))
    return result,False


def run(resume=False):
    for name in ('tables','figures','reports'):(DEST/name).mkdir(parents=True,exist_ok=True)
    fp=fingerprint(); all_rows={k:[] for k in ('queries','partial','events','resources')}
    repath=DEST/'tables/replication_summary.json'
    if not(resume and repath.exists() and (DEST/'tables/replication_fingerprint.json').exists()
        and json.loads((DEST/'tables/replication_fingerprint.json').read_text())==fp):
        replicate();dump(DEST/'tables/replication_fingerprint.json',fp)
    freeze_path=DEST/'tables/frozen_thresholds.json'
    old_freeze=json.loads(freeze_path.read_text()) if resume and freeze_path.exists() else None
    if old_freeze is not None and old_freeze['fingerprint']!=fp:
        raise ValueError('frozen threshold source/config mismatch')
    thresholds={}; hit=0
    for n,c in CONDITIONS:
        cases=[]
        for seed in CAL_SEEDS:
            result,cached=cache_case(seed,n,c,'calibration',fp,None,resume);hit+=cached
            for k in all_rows:all_rows[k].extend(result[k])
            cases.extend(result['queries'])
        frame=pd.DataFrame(cases)
        thresholds[condition_id(n,c)]={arm:float(np.quantile(
            frame[(frame.arm==arm)&(frame.kind=='target')].score,.1,method='linear'))
            for arm in ARMS}
        print(f'校准完成：{condition_id(n,c)}，20种子',flush=True)
    if old_freeze is not None:
        if old_freeze['thresholds']!=thresholds:raise ValueError('threshold reconstruction failed')
        freeze=old_freeze
    else:
        freeze=dict(frozen_at=datetime.now(timezone.utc).isoformat(),
            fingerprint=fp,thresholds=thresholds,method='calibration target q.1 linear')
        dump(freeze_path,freeze)
    # Freeze is persisted before the first confirmation call.
    freeze_hash=sha(freeze_path)
    print('全部阈值已冻结；开始40新种子确认',flush=True)
    for n,c in CONDITIONS:
        t=thresholds[condition_id(n,c)]
        for i,seed in enumerate(CONF_SEEDS):
            result,cached=cache_case(seed,n,c,'confirmation',fp,t,resume);hit+=cached
            for k in all_rows:all_rows[k].extend(result[k])
            if (i+1)%10==0:print(f'确认 {condition_id(n,c)}：{i+1}/40',flush=True)
    if sha(freeze_path)!=freeze_hash:raise ValueError('frozen thresholds changed')
    for k,rows in all_rows.items():pd.DataFrame(rows).to_csv(DEST/'tables'/f'{k}.csv',index=False)
    dump(DEST/'tables/run_record.json',dict(fingerprint=fp,freeze_sha256=freeze_hash,
        cached_cases=hit,total_cases=540,rows={k:len(v) for k,v in all_rows.items()},
        completed_at=datetime.now(timezone.utc).isoformat()))
    summarize();return DEST


def read(name):
    return pd.read_csv(DEST/'tables'/name,float_precision='round_trip')


def summarize():
    q=read('queries.csv');p=read('partial.csv');r=read('resources.csv');e=read('events.csv')
    qc=q[q.cohort=='confirmation'];pc=p[p.cohort=='confirmation']
    rows=[]
    for (seed,cond,arm),g in qc.groupby(['seed','condition','arm']):
        t=g[g.kind=='target'];rec=pc[(pc.seed==seed)&(pc.condition==cond)&(pc.arm==arm)]
        for angle in CONFIG['angles']:
            lure=g[(g.kind=='lure')&(g.theta==angle)]
            vals=np.r_[t.score,lure.score]; labels=np.r_[np.ones(len(t)),np.zeros(len(lure))]
            rows.append(dict(seed=seed,condition=cond,arm=arm,angle=angle,
                auc=roc_auc_score(labels,vals),false_alarm=float(lure.judged_old.mean()),
                hit=float(t.judged_old.mean()),identity=float(rec.correct_identity.mean()),
                mse=float(rec.mse.mean()),convergence=float(rec.converged.mean()),
                outlier_identity=float(rec[rec.parent==g.n_cluster.iloc[0]].correct_identity.mean()),
                cluster_identity=float(rec[rec.parent<g.n_cluster.iloc[0]].correct_identity.mean()),
                unrelated_false_alarm=float(g[g.kind=='unrelated'].judged_old.mean()),
                task2_hit=float(g[g.kind=='task2'].judged_old.mean())))
    metrics=pd.DataFrame(rows);metrics.to_csv(DEST/'tables/metrics_by_seed.csv',index=False)
    group=metrics.groupby(['condition','angle','arm']).mean(numeric_only=True).reset_index()
    group.to_csv(DEST/'tables/group_means.csv',index=False)
    effects=[]
    primary=condition_id(24,.35)
    for (cond,angle),g in metrics.groupby(['condition','angle']):
        for metric in ('identity','false_alarm','auc','hit','mse','cluster_identity','outlier_identity'):
            pivot=g.pivot(index='seed',columns='arm',values=metric)
            for a in ('high_energy','low_energy'):
                z=pivot[a]-pivot.random_replay
                is_primary=cond==primary and angle==.24 and a=='high_energy' and metric in ('identity','false_alarm')
                level=.975 if is_primary else .95
                effects.append(dict(condition=cond,angle=angle,a=a,b='random_replay',metric=metric,
                    n_seeds=len(z),mean=float(z.mean()),ci=bootstrap(z,level=level),level=level,
                    primary=is_primary))
    eff=pd.DataFrame(effects);eff.to_csv(DEST/'tables/paired_effects.csv',index=False)
    chosen=[z for z in effects if z['primary']]
    summary=dict(primary_condition=primary,primary_angle=.24,primary_effects=chosen,
        primary_tradeoff_supported=all(z['ci'][0]>0 for z in chosen),
        replication=json.loads((DEST/'tables/replication_summary.json').read_text()),
        exploratory_matrix_only=True,query_rows=len(q),partial_rows=len(p),event_rows=len(e))
    checks=dict(all_evaluation_read_only=bool(r.query_read_only.all()),
        replay_events_64=bool(r[r.arm!='no_replay'].event_count.eq(64).all()),
        replay_mass_16=bool(r[r.arm!='no_replay'].total_replay_mass.eq(16).all()),
        replay_slots_8=bool(r[r.arm!='no_replay'].buffer_count.eq(8).all()),
        equal_replay_bytes=bool(r[r.arm!='no_replay'].groupby(['seed','condition']).persistent_array_bytes.nunique().eq(1).all()),
        no_new_training_collision=bool(~q[q.kind.isin(['lure','unrelated'])].training_collision.any()),
        frozen_thresholds_match=sha(DEST/'tables/frozen_thresholds.json')==json.loads((DEST/'tables/run_record.json').read_text())['freeze_sha256'],
        no_nonfinite_scores=bool(np.isfinite(q.score).all()),
        convergence_fraction=float(pc.converged.mean()))
    for key,value in checks.items():
        if isinstance(value,bool) and not value:raise RuntimeError(f'audit failed: {key}')
    if len(chosen)!=2:raise RuntimeError('missing primary endpoints')
    # Recompute every recorded judgment from the persisted thresholds.
    freeze=json.loads((DEST/'tables/frozen_thresholds.json').read_text())
    pred=np.array([score>=freeze['thresholds'][cond][arm] for score,cond,arm in
        zip(qc.score,qc.condition,qc.arm)])
    checks['all_judgments_reconstructed']=bool(np.array_equal(pred,qc.judged_old.to_numpy()))
    # Every event must reconstruct the selected source and final masses.
    event_ok=True
    for row in r[r.arm!='no_replay'].itertuples():
        g=e[(e.seed==row.seed)&(e.condition==row.condition)&(e.arm==row.arm)]
        ids=json.loads(row.buffer_ids)
        event_ok &= g.event.tolist()==list(range(64))
        event_ok &= all(ids[int(s)]==int(i) for s,i in zip(g.slot,g.parent))
        event_ok &= np.allclose(g.groupby('slot').gain.sum().to_numpy(),np.full(8,2.),rtol=0,atol=0)
        event_ok &= g.groupby('slot').size().eq(8).all()
    checks['events_reconstruct_buffers_and_masses']=bool(event_ok)
    if not(checks['all_judgments_reconstructed'] and event_ok):raise RuntimeError('reconstruction audit failed')
    dump(DEST/'tables/audit.json',checks);dump(DEST/'tables/summary.json',summary)
    return summary
