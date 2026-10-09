"""Stage 9 replay-source controls; prospectively specified in fidelity_protocol.md."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .energy import Memory, fixture, build_model, CONFIG as ENERGY_CONFIG

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/fidelity'
SOURCES=('exact','coherent','independent')
ARMS=('random_replay','high_energy','low_energy')
READOUTS=('weighted','max_similarity')
GROUPS=(('none','no_replay'),)+tuple((s,a) for s in SOURCES for a in ARMS)
CAL_SEEDS=list(range(20273001,20273021))
CONF_SEEDS=list(range(20274001,20274041))
CONFIG=dict(n=24,c=.35,primary_angle=.24,readouts=READOUTS,
    sources=SOURCES,arms=ARMS,calibration_seeds=CAL_SEEDS,confirmation_seeds=CONF_SEEDS,
    threshold_quantile=.1,bootstrap=50000,bootstrap_seed=2026101009,
    primary_intervals=14,hit_margin=.01,identity_margin=.0025,task2_margin=.01,
    independent_rng_tag=617,energy_configuration=ENERGY_CONFIG)


def key(source,arm): return source+'__'+arm
BASE=key('none','no_replay')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def fingerprint():
    paths=('src/energy.py','src/fidelity.py','fidelity_protocol.md')
    files={p:sha(ROOT/p) for p in paths}
    payload=json.dumps(dict(files=files,configuration=CONFIG),sort_keys=True).encode()
    return dict(sha256=hashlib.sha256(payload).hexdigest(),files=files,configuration=CONFIG)


def source_vectors(f,seed):
    x,y=f['x'],f['y']
    cosine=np.clip(np.sum(x*y,axis=1),-1.,1.)
    rng=np.random.default_rng(np.random.SeedSequence([seed,CONFIG['n'],round(CONFIG['c']*1000),9,CONFIG['independent_rng_tag']]))
    u=rng.normal(size=x.shape)
    u-=np.sum(u*x,axis=1)[:,None]*x
    u/=np.linalg.norm(u,axis=1)[:,None]
    independent=cosine[:,None]*x+np.sqrt(1-cosine**2)[:,None]*u
    return dict(exact=x.copy(),coherent=y.copy(),independent=independent)


def construct(f,source,arm,sources):
    original,ids,events,_=build_model(f,arm)
    vectors=(f['y'].copy() if arm=='no_replay' else np.vstack((f['y'],sources[source][ids])))
    return Memory(vectors,original.masses.copy(),original.beta),ids,events


def recognize(model,query):
    """No labels, old-target dictionary, or source provenance enter recognition."""
    q=np.atleast_2d(query)
    return np.array([-model.energy(q),(q@model.vectors.T).max(axis=1)])


def evaluate_case(seed):
    f=fixture(seed,CONFIG['n'],CONFIG['c']);sources=source_vectors(f,seed)
    arrays={k:f[k] for k in ('x','y','v','query','cues','old_energy','priority','schedule')}
    arrays.update({s+'_all_source_vectors':z for s,z in sources.items()})
    arrays['source_cosines']=np.sum(f['x']*f['y'],axis=1)
    meta=dict(seed=seed,info=f['info'],cue_info=f['cue_info'],groups={})
    for source,arm in GROUPS:
        model,ids,events=construct(f,source,arm,sources)
        name=key(source,arm);before=model.fingerprint();tick=time.perf_counter()
        scores=recognize(model,f['query'])
        part,steps,residual,converged=model.recall(f['cues'],
            tol=ENERGY_CONFIG['tol'],max_iter=ENERGY_CONFIG['max_iter'])
        elapsed=time.perf_counter()-tick
        parents=np.array([r['parent'] for r in f['cue_info']])
        nearest=(part@f['x'].T).argmax(axis=1)
        distances=np.linalg.norm(f['query'][:,None,:]-model.vectors[None,:,:],axis=2).min(axis=1)
        gram=model.vectors@model.vectors.T
        upper=gram[np.triu_indices(len(gram),1)]
        original,original_ids,original_events,_=build_model(f,arm)
        coherent_error=0.
        if source=='coherent':
            weights=np.ones(len(f['y']));weights[ids]+=2.
            collapsed=Memory(f['y'],weights,model.beta)
            coherent_error=max(float(np.abs(collapsed.energy(f['query'])-model.energy(f['query'])).max()),
                               float(np.abs(collapsed.update(f['cues'])-model.update(f['cues'])).max()))
        values=dict(vectors=model.vectors,masses=model.masses,ids=ids,scores=scores,
            partial=part,partial_iterations=steps,partial_residual=residual,
            partial_converged=converged,partial_nearest=nearest,
            partial_correct=(nearest==parents)&converged,
            partial_mse=np.mean((part-f['x'][parents])**2,axis=1),
            min_training_distance=distances)
        arrays.update({name+'_'+k:v for k,v in values.items()})
        exact_matches=bool(np.array_equal(model.vectors,original.vectors) and
            np.array_equal(model.masses,original.masses) and np.array_equal(ids,original_ids)
            and events==original_events)
        meta['groups'][name]=dict(source=source,arm=arm,events=events,buffer_ids=ids.tolist(),
            query_read_only=before==model.fingerprint(),model_hash=before,
            memory_array_bytes=model.vectors.nbytes+model.masses.nbytes,
            recognition_and_partial_seconds=elapsed,
            independent_support_count=int(np.unique(model.vectors,axis=0).shape[0]),
            selected_source_cosines=(np.sum(f['x'][ids]*model.vectors[25:],axis=1).tolist() if len(ids) else []),
            source_gram_rmse=(float(np.sqrt(np.mean((sources[source][ids]@sources[source][ids].T-f['x'][ids]@f['x'][ids].T)**2))) if len(ids) else 0.),
            stored_pairwise_cosine_mean=float(upper.mean()),
            unit_norm_max_error=float(np.abs(np.linalg.norm(model.vectors,axis=1)-1).max()),
            coherent_collapsed_max_error=coherent_error,
            exact_matches_original=(exact_matches if source in ('none','exact') else None))
    return arrays,meta


def cached_case(seed,cohort,fp,resume=False):
    directory=ROOT/'work/fidelity/cases';directory.mkdir(parents=True,exist_ok=True)
    name=f'{cohort}_{seed}';npz=directory/(name+'.npz');record=directory/(name+'.json')
    if resume and record.exists() and npz.exists():
        meta=json.loads(record.read_text())
        if meta['fingerprint_sha256']!=fp['sha256'] or meta['npz_sha256']!=sha(npz):
            raise ValueError('cache fingerprint or raw data mismatch')
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
    return {key(s,a):{readout:float(np.quantile(np.concatenate([
        arrays[key(s,a)+'_scores'][ri,:25] for arrays,_ in cases]),.1,method='linear'))
        for ri,readout in enumerate(READOUTS)} for s,a in GROUPS}


def emit_tables(cases,thresholds):
    rows={n:[] for n in ('queries','partial','events','resources','geometry')}
    audit=dict(query_read_only=True,exact_matches_original=True,
        selected_ids_and_events_equal=True,matched_memory_bytes=True,event_budget_correct=True,
        expected_support_counts=True,nonfinite_scores=0,novel_training_collisions=0,
        independent_cosine_max_error=0.,unit_norm_max_error=0.,
        coherent_max_similarity_error=0.,coherent_max_judgment_disagreements=0,
        coherent_collapsed_max_error=0.,coherent_thresholds_equal=True,
        partial_nonconvergence_count=0)
    for cohort,arrays,meta in cases:
        seed=meta['seed']
        delta=np.abs(np.sum(arrays['x']*arrays['independent_all_source_vectors'],axis=1)-arrays['source_cosines'])
        audit['independent_cosine_max_error']=max(audit['independent_cosine_max_error'],float(delta.max()))
        for s,a in GROUPS:
            name=key(s,a);resource=meta['groups'][name];ids=arrays[name+'_ids']
            scores=arrays[name+'_scores'];masses=arrays[name+'_masses'];ev=resource['events']
            audit['query_read_only'] &= resource['query_read_only']
            audit['nonfinite_scores']+=int((~np.isfinite(scores)).sum())
            audit['unit_norm_max_error']=max(audit['unit_norm_max_error'],resource['unit_norm_max_error'])
            audit['coherent_collapsed_max_error']=max(audit['coherent_collapsed_max_error'],resource['coherent_collapsed_max_error'])
            audit['partial_nonconvergence_count']+=int((~arrays[name+'_partial_converged']).sum())
            if s in ('none','exact'):audit['exact_matches_original'] &= resource['exact_matches_original']
            support=25 if s in ('none','coherent') else 33
            audit['expected_support_counts'] &= resource['independent_support_count']==support
            if a!='no_replay':
                er=meta['groups'][key('exact',a)]
                audit['selected_ids_and_events_equal'] &= (resource['buffer_ids']==er['buffer_ids'] and ev==er['events'])
                audit['matched_memory_bytes'] &= resource['memory_array_bytes']==er['memory_array_bytes']
                counts=np.bincount([e['slot'] for e in ev],minlength=8)
                replay=np.bincount([e['slot'] for e in ev],weights=[e['gain'] for e in ev],minlength=8)
                audit['event_budget_correct'] &= bool(len(ev)==64 and len(ids)==8 and
                    np.array_equal(counts,np.full(8,8)) and np.array_equal(replay,masses[25:])
                    and replay.sum()==16. and all(ids[e['slot']]==e['parent'] for e in ev))
            else:audit['event_budget_correct'] &= len(ev)==0 and len(ids)==0
            if s=='coherent':
                bs=arrays[BASE+'_scores'][1]
                audit['coherent_max_similarity_error']=max(audit['coherent_max_similarity_error'],float(np.abs(bs-scores[1]).max()))
                audit['coherent_thresholds_equal'] &= thresholds[name]['max_similarity']==thresholds[BASE]['max_similarity']
                audit['coherent_max_judgment_disagreements']+=int(((scores[1]>=thresholds[name]['max_similarity'])!=(bs>=thresholds[BASE]['max_similarity'])).sum())
            idset=set(ids.tolist())
            for qi,info in enumerate(meta['info']):
                distance=float(arrays[name+'_min_training_distance'][qi]);collision=distance<1e-6
                novel=info['kind'] in ('lure','unrelated')
                if novel and collision:audit['novel_training_collisions']+=1
                common=dict(seed=seed,cohort=cohort,source=s,arm=a,**info,
                    actual_min_training_distance=distance,actual_training_collision=collision,
                    parent_buffered=info['parent'] in idset,
                    parent_group='outlier' if info['parent']==24 else 'cluster' if info['parent']>=0 else 'none')
                for ri,readout in enumerate(READOUTS):
                    rows['queries'].append(dict(**common,readout=readout,score=float(scores[ri,qi]),
                        judged_old=bool(scores[ri,qi]>=thresholds[name][readout])))
            for qi,info in enumerate(meta['cue_info']):
                rows['partial'].append(dict(seed=seed,cohort=cohort,source=s,arm=a,**info,
                    parent_buffered=info['parent'] in idset,parent_group='outlier' if info['parent']==24 else 'cluster',
                    correct_identity=bool(arrays[name+'_partial_correct'][qi]),
                    recalled_id=int(arrays[name+'_partial_nearest'][qi]),
                    mse=float(arrays[name+'_partial_mse'][qi]),
                    converged=bool(arrays[name+'_partial_converged'][qi]),
                    iterations=int(arrays[name+'_partial_iterations'][qi]),
                    fixed_point_residual=float(arrays[name+'_partial_residual'][qi])))
            rows['events'].extend(dict(seed=seed,cohort=cohort,source=s,arm=a,**e) for e in ev)
            rows['resources'].append(dict(seed=seed,cohort=cohort,
                **{k:v for k,v in resource.items() if k!='events'}))
            for slot,i in enumerate(ids):
                rows['geometry'].append(dict(seed=seed,cohort=cohort,source=s,arm=a,parent=int(i),slot=slot,
                    replay_cosine=float(np.sum(arrays['x'][i]*arrays[name+'_vectors'][25+slot])),
                    matched_base_cosine=float(arrays['source_cosines'][i]),
                    replay_mass=float(masses[25+slot])))
    for name,data in rows.items():pd.DataFrame(data).to_csv(DEST/'tables'/f'{name}.csv',index=False)
    audit.update(cases=len(cases),rows={n:len(data) for n,data in rows.items()})
    required=('query_read_only','exact_matches_original','selected_ids_and_events_equal','matched_memory_bytes',
              'event_budget_correct','expected_support_counts','coherent_thresholds_equal')
    errors=('independent_cosine_max_error','unit_norm_max_error','coherent_max_similarity_error','coherent_collapsed_max_error')
    audit['all_required_checks_pass']=bool(all(audit[n] for n in required) and all(audit[n]<1e-12 for n in errors)
        and audit['nonfinite_scores']==0 and audit['novel_training_collisions']==0
        and audit['coherent_max_judgment_disagreements']==0)
    dump(DEST/'tables/audit.json',audit)
    if not audit['all_required_checks_pass']:raise AssertionError('production audit failed; inspect audit.json')


def paired_ci(values,level):
    v=np.asarray(values,float)
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    res=v[rng.integers(0,len(v),(CONFIG['bootstrap'],len(v)))].mean(axis=1)
    alpha=(1-level)/2
    return float(v.mean()),np.quantile(res,[alpha,1-alpha]).tolist()


def summarize():
    q=pd.read_csv(DEST/'tables/queries.csv',float_precision='round_trip');q=q[q.cohort=='confirmation']
    p=pd.read_csv(DEST/'tables/partial.csv',float_precision='round_trip');p=p[p.cohort=='confirmation']
    metrics=[];subgroups=[]
    for (seed,s,a,r),g in q.groupby(['seed','source','arm','readout'],sort=False):
        t=g[g.kind=='target'];part=p[(p.seed==seed)&(p.source==s)&(p.arm==a)]
        for angle in ENERGY_CONFIG['angles']:
            lure=g[(g.kind=='lure')&(g.theta==angle)]
            metrics.append(dict(seed=seed,source=s,arm=a,readout=r,angle=angle,
                auc=roc_auc_score(np.r_[np.ones(len(t)),np.zeros(len(lure))],np.r_[t.score,lure.score]),
                false_alarm=float(lure.judged_old.mean()),hit=float(t.judged_old.mean()),
                identity=float(part.correct_identity.mean()),partial_mse=float(part.mse.mean()),
                convergence=float(part.converged.mean()),
                task2_hit=float(g[g.kind=='task2'].judged_old.mean()),
                unrelated_false_alarm=float(g[g.kind=='unrelated'].judged_old.mean())))
        for field in ('parent_buffered','parent_group'):
            for value in t[field].unique():
                tg=t[t[field]==value];lg=g[(g.kind=='lure')&(g.theta==.24)&(g[field]==value)]
                pg=part[part[field]==value]
                subgroups.append(dict(seed=seed,source=s,arm=a,readout=r,field=field,value=value,
                    target_count=len(tg),lure_count=len(lg),partial_count=len(pg),
                    hit=float(tg.judged_old.mean()),false_alarm=float(lg.judged_old.mean()),
                    identity=float(pg.correct_identity.mean()),wrong_lures=int(lg.judged_old.sum()),
                    target_score=float(tg.score.mean()),lure_score=float(lg.score.mean())))
    frame=pd.DataFrame(metrics);frame.to_csv(DEST/'tables/metrics_by_seed.csv',index=False)
    group=frame.groupby(['source','arm','readout','angle']).mean(numeric_only=True).drop(columns='seed').reset_index()
    group.to_csv(DEST/'tables/group_means.csv',index=False)
    sub=pd.DataFrame(subgroups);sub.to_csv(DEST/'tables/subgroup_by_seed.csv',index=False)
    sub.groupby(['source','arm','readout','field','value']).mean(numeric_only=True).drop(columns='seed').reset_index().to_csv(DEST/'tables/subgroup_means.csv',index=False)
    indexed={(s,a,r,angle):g.set_index('seed').sort_index() for (s,a,r,angle),g in frame.groupby(['source','arm','readout','angle'])}
    effects=[]
    def add(comparison,s,r,angle,metric,values,primary):
        level=1-.05/14 if primary else .95
        mean,ci=paired_ci(values,level)
        effects.append(dict(comparison=comparison,source=s,readout=r,angle=angle,metric=metric,
            mean=mean,low=ci[0],high=ci[1],confidence=level,primary=primary,seeds=len(values)))
    for r in READOUTS:
        for angle in ENERGY_CONFIG['angles']:
            ex=indexed[('exact','high_energy',r,angle)]
            for s in ('independent','coherent'):
                alt=indexed[(s,'high_energy',r,angle)]
                for metric in ('auc','false_alarm','hit','identity','task2_hit'):
                    primary=s=='independent' and r=='weighted' and angle==.24
                    add('source_minus_exact',s,r,angle,metric,alt[metric]-ex[metric],primary)
            for s in SOURCES:
                hi=indexed[(s,'high_energy',r,angle)];rand=indexed[(s,'random_replay',r,angle)]
                for metric in ('auc','false_alarm','identity'):
                    primary=s in ('exact','independent') and r=='weighted' and angle==.24
                    add('high_minus_random',s,r,angle,metric,hi[metric]-rand[metric],primary)
            for metric in ('auc','false_alarm','identity'):
                independent=indexed[('independent','high_energy',r,angle)][metric]-indexed[('independent','random_replay',r,angle)][metric]
                exact=indexed[('exact','high_energy',r,angle)][metric]-indexed[('exact','random_replay',r,angle)][metric]
                add('policy_interaction_independent_minus_exact','independent',r,angle,metric,independent-exact,r=='weighted' and angle==.24)
    eff=pd.DataFrame(effects);eff.to_csv(DEST/'tables/paired_effects.csv',index=False)
    prim=eff[eff.primary];assert len(prim)==14
    direct=prim[prim.comparison=='source_minus_exact'].set_index('metric')
    decisions=dict(auc_improvement=bool(direct.loc['auc','low']>0),
        false_alarm_reduction=bool(direct.loc['false_alarm','high']<0),
        hit_noninferior=bool(direct.loc['hit','low']>=-.01),
        identity_noninferior=bool(direct.loc['identity','low']>=-.0025),
        task2_noninferior=bool(direct.loc['task2_hit','low']>=-.01))
    decisions['joint_pass']=all(decisions.values())
    diagnosis=[]
    for s,a in GROUPS:
        for r in READOUTS:
            g=q[(q.source==s)&(q.arm==a)&(q.readout==r)&(q.kind=='lure')&(q.theta==.24)]
            buffered=g[g.parent_buffered];outside=g[~g.parent_buffered]
            wrong=int(g.judged_old.sum());bufwrong=int(buffered.judged_old.sum())
            diagnosis.append(dict(source=s,arm=a,readout=r,lures=len(g),wrong=wrong,
                buffered_lures=len(buffered),buffered_wrong=bufwrong,
                buffered_false_alarm=float(buffered.judged_old.mean()) if len(buffered) else None,
                outside_lures=len(outside),outside_wrong=int(outside.judged_old.sum()),
                outside_false_alarm=float(outside.judged_old.mean()),
                fraction_errors_buffered=bufwrong/wrong if wrong else None))
    dump(DEST/'tables/mechanism_diagnostic.json',dict(main_angle=.24,counts=diagnosis,
        limitation='改变源精度同时改变局部几何，分层只作诊断；不能声称唯一因果或人脑机制'))
    dump(DEST/'tables/summary.json',dict(primary_readout='weighted',primary_angle=.24,
        confirmation_seeds=40,primary_intervals=14,confidence=1-.05/14,
        primary_means=group[(group.arm=='high_energy')&(group.readout=='weighted')&(group.angle==.24)].to_dict('records'),
        primary_effects=prim.to_dict('records'),decisions=decisions,bootstrap=CONFIG['bootstrap']))


def run(resume=False):
    for folder in ('tables','figures','reports'):(DEST/folder).mkdir(parents=True,exist_ok=True)
    fp=fingerprint();freeze_path=DEST/'tables/frozen_thresholds.json'
    if freeze_path.exists() and not resume:raise ValueError('results already frozen; use --resume or a new export')
    old=json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    if old and old['fingerprint']['sha256']!=fp['sha256']:raise ValueError('frozen source/config mismatch')
    cases=[];calibration=[];hits=0
    for i,seed in enumerate(CAL_SEEDS):
        arrays,meta,reused=cached_case(seed,'calibration',fp,resume)
        hits+=int(reused);calibration.append((arrays,meta));cases.append(('calibration',arrays,meta))
        if (i+1)%5==0:print(f'校准 {i+1}/20；缓存命中{hits}',flush=True)
    thresholds=calibrate(calibration)
    if old:
        if old['thresholds']!=thresholds:raise AssertionError('threshold reproduction failed')
    else:dump(freeze_path,dict(created_utc=datetime.now(timezone.utc).isoformat(),fingerprint=fp,
        thresholds=thresholds,method='500 calibration old targets per group/readout; q.1 linear'))
    freeze_hash=sha(freeze_path);print('阈值已冻结，开始确认：'+freeze_hash,flush=True)
    for i,seed in enumerate(CONF_SEEDS):
        arrays,meta,reused=cached_case(seed,'confirmation',fp,resume)
        hits+=int(reused);cases.append(('confirmation',arrays,meta))
        if (i+1)%5==0:print(f'确认 {i+1}/40；缓存命中{hits}',flush=True)
    if sha(freeze_path)!=freeze_hash:raise AssertionError('frozen threshold changed')
    emit_tables(cases,thresholds);summarize()
    record=dict(fingerprint=fp,freeze_sha256=freeze_hash,cached_cases=hits,total_cases=len(cases),
        finished_utc=datetime.now(timezone.utc).isoformat())
    dump(DEST/'tables/run_record.json',record)
    if not (DEST/'tables/initial_run_record.json').exists():dump(DEST/'tables/initial_run_record.json',record)
    print('第九阶段计算与审核完成',flush=True)
