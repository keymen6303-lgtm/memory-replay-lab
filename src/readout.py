"""Stage 8: fixed-memory recognition readout diagnostic; see readout_protocol.md."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .energy import Memory, fixture, build_model, ARMS, CONFIG as ENERGY_CONFIG

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'outputs/readout'
READOUTS = ('weighted', 'equal', 'reconstruction', 'max_similarity')
CAL_SEEDS = list(range(20271001, 20271021))
CONF_SEEDS = list(range(20272001, 20272041))
CONFIG = dict(n=24, c=.35, primary_angle=.24, beta=8.,
              readouts=READOUTS, calibration_seeds=CAL_SEEDS,
              confirmation_seeds=CONF_SEEDS, threshold_quantile=.1,
              bootstrap=50000, bootstrap_seed=2026101008,
              primary_intervals=9, noninferiority_margin=.01,
              energy_configuration=ENERGY_CONFIG)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def fingerprint():
    paths=('src/energy.py','src/readout.py','readout_protocol.md')
    source={p:sha(ROOT/p) for p in paths}
    encoded=json.dumps(dict(source=source, config=CONFIG), sort_keys=True).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), sources=source, config=CONFIG)


def readouts(model, query):
    """Receives only actual model and unlabeled queries; leaves model immutable."""
    before=model.fingerprint(); q=np.atleast_2d(query)
    full, iterations, residual, converged=model.recall(q,
        tol=ENERGY_CONFIG['tol'], max_iter=ENERGY_CONFIG['max_iter'])
    scores=np.array([
        -model.energy(q),
        -Memory(model.vectors, np.ones(len(model.vectors)), model.beta).energy(q),
        -np.mean((full-q)**2, axis=1),
        (q@model.vectors.T).max(axis=1),
    ])
    if before != model.fingerprint():
        raise AssertionError('recognition mutated model')
    return scores, full, iterations, residual, converged


def decomposition(model, query, base_count):
    q=np.atleast_2d(query)
    base=Memory(model.vectors[:base_count],np.ones(base_count),model.beta)
    equal=Memory(model.vectors,np.ones(len(model.vectors)),model.beta)
    s0=-base.energy(q); s1=-equal.energy(q); s2=-model.energy(q)
    return s0,s1-s0,s2-s1,s2-s0


def evaluate_case(seed):
    f=fixture(seed,CONFIG['n'],CONFIG['c'])
    arrays={k:f[k] for k in ('x','y','v','query','cues','old_energy','priority','schedule')}
    meta=dict(seed=seed, info=f['info'], cue_info=f['cue_info'], arms={})
    for arm in ARMS:
        model,ids,events,_=build_model(f,arm)
        before=model.fingerprint(); tick=time.perf_counter()
        scores, full, iterations, residual, converged=readouts(model,f['query'])
        elapsed=time.perf_counter()-tick
        partial, psteps, presidual, pok=model.recall(f['cues'],
            tol=ENERGY_CONFIG['tol'],max_iter=ENERGY_CONFIG['max_iter'])
        parents=np.array([row['parent'] for row in f['cue_info']])
        nearest=(partial@f['x'].T).argmax(axis=1)
        partial_correct=(nearest==parents)&pok
        pmse=np.mean((partial-f['x'][parents])**2,axis=1)
        total_mass=model.masses.sum()
        normalized=Memory(model.vectors,model.masses/total_mass,model.beta)
        norm_scores=-normalized.energy(f['query'])
        shift=np.log(total_mass)/model.beta
        norm_error=float(np.max(np.abs(norm_scores-(scores[0]-shift))))
        update_error=float(np.max(np.abs(normalized.update(f['query'])-model.update(f['query']))))
        s0, support, weight, total=decomposition(model,f['query'],len(f['y']))
        decomp_error=float(np.max(np.abs(total-support-weight)))
        prefix=arm+'_'
        for key,value in dict(vectors=model.vectors,masses=model.masses,ids=ids,
            scores=scores,full=full,full_iterations=iterations,full_residual=residual,
            full_converged=converged,partial=partial,partial_iterations=psteps,
            partial_residual=presidual,partial_converged=pok,partial_correct=partial_correct,
            partial_mse=pmse,partial_nearest=nearest,normalized_scores=norm_scores,
            base_score=s0,support_contribution=support,weight_contribution=weight,
            total_contribution=total).items():
            arrays[prefix+key]=value
        meta['arms'][arm]=dict(events=events, buffer_ids=ids.tolist(),
            query_read_only=before==model.fingerprint(), model_hash=before,
            memory_array_bytes=model.vectors.nbytes+model.masses.nbytes,
            recognition_seconds=elapsed, norm_shift=float(shift),
            normalization_max_error=norm_error,normalized_update_max_error=update_error,
            decomposition_max_error=decomp_error)
    return arrays,meta


def cached_case(seed,cohort,fp,resume=False):
    directory=ROOT/'work/readout/cases'; directory.mkdir(parents=True,exist_ok=True)
    name=f'{cohort}_{seed}'; npz=directory/(name+'.npz'); record=directory/(name+'.json')
    if resume and record.exists() and npz.exists():
        meta=json.loads(record.read_text())
        if meta['fingerprint_sha256']!=fp['sha256'] or meta['npz_sha256']!=sha(npz):
            raise ValueError('cache fingerprint or data hash mismatch')
        with np.load(npz,allow_pickle=False) as loaded:
            arrays={k:loaded[k] for k in loaded.files}
        return arrays,meta,True
    arrays,meta=evaluate_case(seed)
    np.savez_compressed(npz,**arrays)
    meta.update(cohort=cohort,fingerprint_sha256=fp['sha256'],npz_sha256=sha(npz),
                created_utc=datetime.now(timezone.utc).isoformat())
    dump(record,meta)
    return arrays,meta,False


def calibrate(cases):
    thresholds={}
    for arm in ARMS:
        thresholds[arm]={}
        for ri,name in enumerate(READOUTS):
            targets=np.concatenate([a[arm+'_scores'][ri,:CONFIG['n']+1] for a,_ in cases])
            thresholds[arm][name]=float(np.quantile(targets,.1,method='linear'))
    return thresholds


def emit_tables(all_cases,thresholds):
    queries=[]; partial=[]; decomp=[]; resources=[]; events=[]
    checks=dict(query_read_only=True,novel_training_collision_count=0,
        normalization_max_error=0.,normalized_update_max_error=0.,decomposition_max_error=0.,
        normalized_judgment_disagreements=0,normalized_auc_max_error=0.,
        event_budget_correct=True,stored_vector_unit_max_error=0.,
        nonfinite_score_count=0,baseline_score_matches_no_replay=True)
    for cohort,arrays,meta in all_cases:
        seed=meta['seed']; info=meta['info']; m=CONFIG['n']+1
        for row in info:
            if row['kind'] in ('lure','unrelated') and row['training_collision']:
                checks['novel_training_collision_count']+=1
        for arm in ARMS:
            prefix=arm+'_'; ids=set(meta['arms'][arm]['buffer_ids']); resource=meta['arms'][arm]
            checks['query_read_only'] &= resource['query_read_only']
            for name in ('normalization_max_error','normalized_update_max_error','decomposition_max_error'):
                checks[name]=max(checks[name],resource[name])
            err=np.max(np.abs(np.linalg.norm(arrays[prefix+'vectors'],axis=1)-1))
            checks['stored_vector_unit_max_error']=max(checks['stored_vector_unit_max_error'],float(err))
            scores=arrays[prefix+'scores']; ns=arrays[prefix+'normalized_scores']
            checks['nonfinite_score_count']+=int((~np.isfinite(scores)).sum())
            pred=scores[0]>=thresholds[arm]['weighted']
            normpred=ns>=(thresholds[arm]['weighted']-resource['norm_shift'])
            checks['normalized_judgment_disagreements']+=int((pred!=normpred).sum())
            for angle in ENERGY_CONFIG['angles']:
                relevant=[i for i,r in enumerate(info) if r['kind']=='target' or
                          (r['kind']=='lure' and r['theta']==angle)]
                labels=[info[i]['kind']=='target' for i in relevant]
                diff=abs(roc_auc_score(labels,scores[0,relevant])-roc_auc_score(labels,ns[relevant]))
                checks['normalized_auc_max_error']=max(checks['normalized_auc_max_error'],diff)
            checks['baseline_score_matches_no_replay'] &= bool(np.allclose(
                arrays[prefix+'base_score'],arrays['no_replay_scores'][0],rtol=0,atol=1e-12))
            masses=arrays[prefix+'masses']; ev=resource['events']
            if arm=='no_replay':
                checks['event_budget_correct'] &= len(ev)==0 and len(ids)==0
            else:
                counts=np.bincount([e['slot'] for e in ev],minlength=8)
                reconstructed=np.bincount([e['slot'] for e in ev],
                    weights=[e['gain'] for e in ev],minlength=8)
                checks['event_budget_correct'] &= bool(len(ev)==64 and len(ids)==8 and
                    np.array_equal(counts,np.full(8,8)) and
                    np.array_equal(reconstructed,masses[m:]) and masses[m:].sum()==16 and
                    all(resource['buffer_ids'][e['slot']]==e['parent'] for e in ev))
            for qi,row in enumerate(info):
                common=dict(seed=seed,cohort=cohort,arm=arm,**row,
                    parent_group=('outlier' if row['parent']==CONFIG['n'] else
                                  'cluster' if row['parent']>=0 else 'none'),
                    parent_buffered=row['parent'] in ids)
                for ri,name in enumerate(READOUTS):
                    queries.append(dict(**common,readout=name,score=float(scores[ri,qi]),
                        judged_old=bool(scores[ri,qi]>=thresholds[arm][name]),
                        full_recall_converged=bool(arrays[prefix+'full_converged'][qi])))
                decomp.append(dict(**common,base_score=float(arrays[prefix+'base_score'][qi]),
                    support_contribution=float(arrays[prefix+'support_contribution'][qi]),
                    weight_contribution=float(arrays[prefix+'weight_contribution'][qi]),
                    total_contribution=float(arrays[prefix+'total_contribution'][qi])))
            for qi,row in enumerate(meta['cue_info']):
                partial.append(dict(seed=seed,cohort=cohort,arm=arm,**row,
                    parent_group='outlier' if row['parent']==CONFIG['n'] else 'cluster',
                    parent_buffered=row['parent'] in ids,
                    correct_identity=bool(arrays[prefix+'partial_correct'][qi]),
                    recalled_id=int(arrays[prefix+'partial_nearest'][qi]),
                    mse=float(arrays[prefix+'partial_mse'][qi]),
                    converged=bool(arrays[prefix+'partial_converged'][qi]),
                    iterations=int(arrays[prefix+'partial_iterations'][qi]),
                    fixed_point_residual=float(arrays[prefix+'partial_residual'][qi])))
            resources.append(dict(seed=seed,cohort=cohort,arm=arm,
                **{k:v for k,v in resource.items() if k!='events'},
                full_recall_converged_fraction=float(arrays[prefix+'full_converged'].mean())))
            events.extend(dict(seed=seed,cohort=cohort,arm=arm,**e) for e in ev)
    for name,rows in dict(queries=queries,partial=partial,decomposition=decomp,
                           resources=resources,events=events).items():
        pd.DataFrame(rows).to_csv(DEST/'tables'/f'{name}.csv',index=False)
    checks['all_required_checks_pass']=bool(checks['query_read_only'] and
        checks['novel_training_collision_count']==0 and checks['normalization_max_error']<1e-12 and
        checks['normalized_update_max_error']<1e-12 and checks['decomposition_max_error']<1e-12 and
        checks['normalized_judgment_disagreements']==0 and checks['normalized_auc_max_error']==0 and
        checks['event_budget_correct'] and checks['stored_vector_unit_max_error']<1e-12 and
        checks['nonfinite_score_count']==0 and checks['baseline_score_matches_no_replay'])
    checks.update(cases=len(all_cases),query_rows=len(queries),partial_rows=len(partial),
                  decomposition_rows=len(decomp),event_rows=len(events))
    dump(DEST/'tables/audit.json',checks)
    if not checks['all_required_checks_pass']:
        raise AssertionError('production audit failed: inspect audit.json')


def paired_ci(values,level=.95):
    v=np.asarray(values,float)
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    samples=v[rng.integers(0,len(v),size=(CONFIG['bootstrap'],len(v)))].mean(axis=1)
    tails=(1-level)/2
    return float(v.mean()),np.quantile(samples,[tails,1-tails]).tolist()


def summarize():
    q=pd.read_csv(DEST/'tables/queries.csv');q=q[q.cohort=='confirmation']
    p=pd.read_csv(DEST/'tables/partial.csv');p=p[p.cohort=='confirmation']
    metrics=[];subgroups=[]
    for (seed,arm,readout),group in q.groupby(['seed','arm','readout'],sort=False):
        target=group[group.kind=='target']
        unrelated=group[group.kind=='unrelated'];task2=group[group.kind=='task2']
        part=p[(p.seed==seed)&(p.arm==arm)]
        for angle in ENERGY_CONFIG['angles']:
            lure=group[(group.kind=='lure')&(group.theta==angle)]
            auc=roc_auc_score(np.r_[np.ones(len(target)),np.zeros(len(lure))],
                              np.r_[target.score,lure.score])
            metrics.append(dict(seed=seed,arm=arm,readout=readout,angle=angle,
                auc=auc,false_alarm=lure.judged_old.mean(),hit=target.judged_old.mean(),
                unrelated_false_alarm=unrelated.judged_old.mean(),task2_hit=task2.judged_old.mean(),
                identity=part.correct_identity.mean(),partial_mse=part.mse.mean(),
                full_convergence=group.full_recall_converged.mean()))
        for field in ('parent_group','parent_buffered'):
            for value in target[field].unique():
                tg=target[target[field]==value]
                lg=group[(group.kind=='lure')&(group.theta==.24)&(group[field]==value)]
                pg=part[part[field]==value]
                if len(tg) and len(lg):
                    subgroups.append(dict(seed=seed,arm=arm,readout=readout,field=field,
                        value=value,target_count=len(tg),lure_count=len(lg),
                        hit=tg.judged_old.mean(),false_alarm=lg.judged_old.mean(),
                        identity=pg.correct_identity.mean(),
                        target_score=tg.score.mean(),lure_score=lg.score.mean()))
    frame=pd.DataFrame(metrics);frame.to_csv(DEST/'tables/metrics_by_seed.csv',index=False)
    groups=frame.groupby(['arm','readout','angle']).mean(numeric_only=True).drop(columns='seed').reset_index()
    groups.to_csv(DEST/'tables/group_means.csv',index=False)
    pd.DataFrame(subgroups).to_csv(DEST/'tables/subgroup_by_seed.csv',index=False)
    effects=[]
    for arm in ARMS:
        for angle in ENERGY_CONFIG['angles']:
            g=frame[(frame.arm==arm)&(frame.angle==angle)]
            base=g[g.readout=='weighted'].set_index('seed').sort_index()
            for name in READOUTS[1:]:
                alternative=g[g.readout==name].set_index('seed').sort_index()
                for metric in ('auc','false_alarm','hit'):
                    is_primary=arm=='high_energy' and angle==CONFIG['primary_angle']
                    level=1-.05/CONFIG['primary_intervals'] if is_primary else .95
                    differences=alternative[metric]-base[metric]
                    mean,ci=paired_ci(differences,level=level)
                    effects.append(dict(arm=arm,angle=angle,readout=name,metric=metric,
                        mean=mean,low=ci[0],high=ci[1],confidence=level,
                        primary=is_primary,seeds=len(differences)))
    effects_frame=pd.DataFrame(effects)
    effects_frame.to_csv(DEST/'tables/paired_effects.csv',index=False)
    primary=effects_frame[effects_frame.primary]
    decisions=[]
    for name in READOUTS[1:]:
        bymetric=primary[primary.readout==name].set_index('metric')
        decisions.append(dict(readout=name,auc_improvement=bool(bymetric.loc['auc','low']>0),
            false_alarm_reduction=bool(bymetric.loc['false_alarm','high']<0),
            hit_noninferior=bool(bymetric.loc['hit','low']>=-CONFIG['noninferiority_margin']),
            joint_pass=bool(bymetric.loc['auc','low']>0 and
                bymetric.loc['false_alarm','high']<0 and
                bymetric.loc['hit','low']>=-CONFIG['noninferiority_margin'])))
    dec=pd.read_csv(DEST/'tables/decomposition.csv');dec=dec[dec.cohort=='confirmation']
    dec.groupby(['arm','kind','theta']).mean(numeric_only=True)[
        ['support_contribution','weight_contribution','total_contribution']].reset_index().to_csv(
            DEST/'tables/decomposition_means.csv',index=False)
    dump(DEST/'tables/summary.json',dict(primary_arm='high_energy',primary_angle=.24,
        confirmation_seeds=40,primary_means=groups[(groups.arm=='high_energy')&(groups.angle==.24)].to_dict('records'),
        primary_effects=primary.to_dict('records'),decisions=decisions,
        all_primary_intervals=9,bootstrap=CONFIG['bootstrap']))


def run(resume=False):
    for folder in ('tables','figures','reports'):(DEST/folder).mkdir(parents=True,exist_ok=True)
    fp=fingerprint();freeze_path=DEST/'tables/frozen_thresholds.json'
    if freeze_path.exists() and not resume:
        raise ValueError('results already frozen; use --resume or a fresh export directory')
    old=json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    if old and old['fingerprint']['sha256']!=fp['sha256']:
        raise ValueError('frozen source/config mismatch')
    all_cases=[];calibration=[];hits=0
    for index,seed in enumerate(CAL_SEEDS):
        arrays,meta,reused=cached_case(seed,'calibration',fp,resume)
        hits+=int(reused);calibration.append((arrays,meta));all_cases.append(('calibration',arrays,meta))
        if (index+1)%5==0:print(f'calibration {index+1}/20; cache hits {hits}',flush=True)
    thresholds=calibrate(calibration)
    if old:
        if old['thresholds']!=thresholds:raise AssertionError('threshold reproduction failed')
    else:
        dump(freeze_path,dict(created_utc=datetime.now(timezone.utc).isoformat(),
            fingerprint=fp,thresholds=thresholds,method='500 calibration old targets per arm/readout, q.1 linear'))
    freeze_hash=sha(freeze_path)
    print('thresholds frozen before confirmation: '+freeze_hash,flush=True)
    for index,seed in enumerate(CONF_SEEDS):
        arrays,meta,reused=cached_case(seed,'confirmation',fp,resume)
        hits+=int(reused);all_cases.append(('confirmation',arrays,meta))
        if (index+1)%5==0:print(f'confirmation {index+1}/40; cache hits {hits}',flush=True)
    if sha(freeze_path)!=freeze_hash:raise AssertionError('freeze changed')
    emit_tables(all_cases,thresholds);summarize()
    record=dict(fingerprint=fp,freeze_sha256=freeze_hash,cached_cases=hits,
        total_cases=len(all_cases),finished_utc=datetime.now(timezone.utc).isoformat())
    dump(DEST/'tables/run_record.json',record)
    if not (DEST/'tables/initial_run_record.json').exists():
        dump(DEST/'tables/initial_run_record.json',record)
    print('completed stage 8 '+json.dumps(record,ensure_ascii=False),flush=True)
