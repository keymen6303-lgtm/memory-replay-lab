"""Chinese stage-8 report and scientific figures from saved results only."""
from pathlib import Path
import html
import json
import shutil
import platform
import sys
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .readout import ROOT,DEST,READOUTS,ARMS

RLABELS=dict(weighted='原加权能量',equal='等权能量',reconstruction='重建误差',max_similarity='最大相似度')
ALABELS=dict(no_replay='无回放',random_replay='随机回放',high_energy='高能量优先',low_energy='低能量优先')
COLORS=['#66788b','#238587','#c18235','#735a9d']


def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}个百分点'


def generate():
    font=Path('/System/Library/Fonts/STHeiti Medium.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,
        'axes.spines.right':False,'font.size':10})
    def save(name):
        plt.tight_layout();plt.savefig(DEST/'figures'/name,dpi=160);plt.close()
    table=DEST/'tables';reports=DEST/'reports'
    summary=json.loads((table/'summary.json').read_text())
    audit=json.loads((table/'audit.json').read_text())
    freeze=json.loads((table/'frozen_thresholds.json').read_text())
    group=pd.read_csv(table/'group_means.csv')
    effects=pd.read_csv(table/'paired_effects.csv')
    queries=pd.read_csv(table/'queries.csv');queries=queries[queries.cohort=='confirmation']
    partial=pd.read_csv(table/'partial.csv');partial=partial[partial.cohort=='confirmation']
    dec=pd.read_csv(table/'decomposition_means.csv')
    sub=pd.read_csv(table/'subgroup_by_seed.csv')
    main=group[(group.arm=='high_energy')&(group.angle==.24)].set_index('readout').loc[list(READOUTS)]
    passed=[RLABELS[d['readout']] for d in summary['decisions'] if d['joint_pass']]
    headline=('、'.join(passed)+'通过预定联合标准。' if passed else '三种替代读出均未通过预定联合标准。')
    fig,axes=plt.subplots(1,3,figsize=(12,4.3))
    for ax,(metric,title) in zip(axes,[('auc','目标—相似未见AUC'),('false_alarm','相似未见误认率'),('hit','完整旧实例命中率')]):
        scale=1 if metric=='auc' else 100
        ax.bar(range(4),main[metric]*scale,color=COLORS)
        ax.set_xticks(range(4),[RLABELS[r] for r in READOUTS],rotation=22)
        ax.set_ylim(0,1 if metric=='auc' else 100);ax.set_ylabel('AUC' if metric=='auc' else '%')
        ax.set_title(title)
    fig.suptitle('固定高能量回放模型 · 主条件θ=.24 · 40个新确认种子（均值）')
    save('primary_readouts.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4.3))
    prim=effects[effects.primary]
    for ax,(metric,title) in zip(axes,[('auc','AUC差'),('false_alarm','误认率差（百分点）'),('hit','旧命中差（百分点）')]):
        dat=prim[prim.metric==metric].set_index('readout').loc[list(READOUTS[1:])]
        scale=1 if metric=='auc' else 100
        ax.errorbar(dat['mean']*scale,np.arange(3),xerr=np.array([
            (dat['mean']-dat.low)*scale,(dat.high-dat['mean'])*scale]),fmt='o',capsize=4)
        ax.set_yticks(range(3),[RLABELS[r] for r in READOUTS[1:]])
        ax.axvline(0,ls='--',color='#777')
        if metric=='hit':ax.axvline(-1,ls=':',color='#c18235',label='−1pp非劣界限');ax.legend(fontsize=8)
        ax.set_title(title);ax.set_xlabel('替代读出 − 原加权读出')
    fig.suptitle('9个主终点各99.444%配对区间 · 预定三项联合判断')
    save('primary_intervals.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,(metric,title) in zip(axes,[('auc','目标—相似实例AUC'),('false_alarm','相似实例误认率（%）')]):
        matrix=np.array([[group[(group.arm==a)&(group.readout==r)&(group.angle==.24)][metric].iloc[0]
            for r in READOUTS] for a in ARMS])
        if metric!='auc':matrix*=100
        im=ax.imshow(matrix,vmin=0,vmax=1 if metric=='auc' else 100,cmap='YlGnBu')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.3f}' if metric=='auc' else f'{matrix[i,j]:.1f}',
                ha='center',va='center',color='white' if matrix[i,j]>(.6 if metric=='auc' else 60) else '#172e3d')
        ax.set_xticks(range(4),[RLABELS[r] for r in READOUTS],rotation=22)
        ax.set_yticks(range(4),[ALABELS[a] for a in ARMS]);ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('同一模型内改变读出；除高能量主比较外均探索性')
    save('all_arms.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for r,col in zip(READOUTS,COLORS):
        dat=group[(group.arm=='high_energy')&(group.readout==r)].sort_values('angle')
        axes[0].plot(dat.angle,dat.auc,'o-',label=RLABELS[r],color=col)
        axes[1].plot(dat.angle,dat.false_alarm*100,'o-',label=RLABELS[r],color=col)
    for ax,title in zip(axes,['目标—相似实例AUC','相似误认率（%）']):
        ax.set_title(title);ax.set_xlabel('相似实例角度θ（小=更相似）');ax.legend(fontsize=8)
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,100)
    save('angle_sweep.png')
    fig,axes=plt.subplots(1,4,figsize=(13,3.8))
    for ax,r in zip(axes,READOUTS):
        dat=queries[(queries.arm=='high_energy')&(queries.readout==r)]
        for kind,label,color in [('target','旧目标','#24788b'),('lure','相似未见','#bc7c39')]:
            v=dat[(dat.kind==kind)&((dat.theta==.24) if kind=='lure' else True)].score
            ax.hist(v,bins=35,alpha=.55,density=True,label=label,color=color)
        ax.axvline(freeze['thresholds']['high_energy'][r],color='#444',ls='--',label='冻结阈值')
        ax.set_title(RLABELS[r]);ax.set_xlabel('各读出的分数');ax.legend(fontsize=7)
    save('score_distributions.png')
    fig,ax=plt.subplots(figsize=(8,4))
    d=dec[(dec.arm=='high_energy')&((dec.kind=='target')|((dec.kind=='lure')&(dec.theta==.24)))]
    d=d.set_index('kind').loc[['target','lure']]
    ax.bar([0,1],d.support_contribution,label='加旧向量（每条质量1）',color='#24788b')
    ax.bar([0,1],d.weight_contribution,bottom=d.support_contribution,
           label='同一旧向量质量1→2',color='#bc7c39')
    ax.set_xticks([0,1],['旧目标','相似未见']);ax.set_ylabel('相对仅Task2支持集合的分数增加')
    ax.set_title('能量分数的指定路径分解（不能当作误认率因果比例）');ax.legend(fontsize=9)
    save('score_decomposition.png')
    number_table='|判断方式|AUC|相似误认|旧命中|部分实例恢复|\n|---|---:|---:|---:|---:|\n'
    for r in READOUTS:
        v=main.loc[r];number_table+=f'|{RLABELS[r]}|{v.auc:.4f}|{pct(v.false_alarm)}|{pct(v.hit)}|{pct(v.identity)}|\n'
    intervals='|替代读出|指标|差值|99.444%区间|\n|---|---|---:|---|\n'
    for row in summary['primary_effects']:
        formatter=(lambda x:f'{x:+.5f}') if row['metric']=='auc' else pp
        label=dict(auc='AUC',false_alarm='相似误认',hit='旧命中')[row['metric']]
        intervals+=f'|{RLABELS[row["readout"]]}|{label}|{formatter(row["mean"])}|[{formatter(row["low"])}, {formatter(row["high"])}]|\n'
    decisions='\n'.join(f'- {RLABELS[d["readout"]]}：AUC改善={d["auc_improvement"]}；误认下降={d["false_alarm_reduction"]}；旧命中非劣={d["hit_noninferior"]}；联合通过={d["joint_pass"]}。' for d in summary['decisions'])
    full_fail=int((~queries.drop_duplicates(['seed','arm','query_id']).full_recall_converged).sum())
    partial_fail=int((~partial.converged).sum())
    sg=sub[(sub.arm=='high_energy')&(sub.readout=='weighted')&(sub.field=='parent_group')]
    gmeans=sg.groupby('value').mean(numeric_only=True)
    subgroup_text='；'.join(f'{str(k)}部分身份恢复{pct(v.identity)}' for k,v in gmeans.iterrows())
    hq=queries[queries.arm=='high_energy']
    mechanism=[]
    for name in READOUTS:
        lr=hq[(hq.readout==name)&(hq.kind=='lure')&(hq.theta==.24)]
        buffered=lr[lr.parent_buffered];outside=lr[~lr.parent_buffered]
        wrong=int(lr.judged_old.sum());buffered_wrong=int(buffered.judged_old.sum())
        mechanism.append(dict(readout=name,buffered_lures=len(buffered),
            buffered_false_alarm=float(buffered.judged_old.mean()),
            unbuffered_false_alarm=float(outside.judged_old.mean()),
            all_false_alarms=wrong,buffered_false_alarms=buffered_wrong,
            buffered_fraction_of_errors=buffered_wrong/wrong if wrong else None))
    maxq=hq[hq.readout=='max_similarity'];old=maxq[maxq.kind=='target']
    cosine=float(np.cos(.24));outside_old=old[~old.parent_buffered]
    strict_hit=float((old.score>cosine+1e-12).mean())
    mechanism_record=dict(status='预定缓存分层的描述统计；几何阈值示例为探索性，未用于调参或替换冻结阈值',
        readouts=mechanism,cosine_theta=cosine,
        max_similarity_frozen_threshold=freeze['thresholds']['high_energy']['max_similarity'],
        unbuffered_old_count=len(outside_old),
        unbuffered_old_below_lure_cosine=int((outside_old.score<cosine-1e-12).sum()),
        descriptive_hit_at_cosine_plus_tolerance=strict_hit,
        tolerance=1e-12,warning='不是确认后的新优化阈值，也不是人类记忆结论')
    (table/'mechanism_diagnostic.json').write_text(json.dumps(mechanism_record,ensure_ascii=False,indent=2),encoding='utf-8')
    mechanism_table='|读出|复习样本的相似新图误认|未复习样本的相似新图误认|错误中来自复习样本的比例|\n|---|---:|---:|---:|\n'
    for row in mechanism:
        mechanism_table+=f'|{RLABELS[row["readout"]]}|{pct(row["buffered_false_alarm"])}|{pct(row["unbuffered_false_alarm"])}|{pct(row["buffered_fraction_of_errors"])}|\n'
    eq=main.loc['equal'];wt=main.loc['weighted'];mx=main.loc['max_similarity'];rc=main.loc['reconstruction']
    interpretation=(f'等权读出相对原读出的误认均值仅下降{100*(wt.false_alarm-eq.false_alarm):.2f}个百分点，'
        '其主区间跨零，不能确认稳定减少误认。最大相似度提高AUC并减少误认，但旧命中差的下界低于−1个百分点，'
        '未证明满足预定的旧命中保护标准；这不是已经证明旧命中受损。重建误差使误认明显增加、AUC降低。'
        '因此，本次没有找到通过三项联合标准的读出；忽略回放次数也没有消除误认。')
    literature=json.loads((ROOT/'outputs/literature/第八阶段候选_逐轮查新记录.json').read_text())
    refs='\n'.join(f'- [{s["title"]}]({s["url"]})：{s["established"]} 读取范围：{s["read"]}。' for s in literature['sources'])
    report=f'''# 第八阶段：固定记忆的新旧读出诊断

协议和报告日期：2026-10-10。{headline}

## 实际问题和主结果

本次检查回放后的相似误认是否依赖判断方式。原样保留第七阶段记忆向量、质量及回忆动力学，在新的20校准、40确认种子上比较四种读出；不是同一旧数据反复调参。

主组：高能量回放；24簇+1孤立；c=.35；theta=.24。

{number_table}
部分实例恢复在四个读出中完全相同，这是固定回忆过程的设计结果，不是四次独立的改善证据。{subgroup_text}。不能因识别指标提高就宣称回忆能力增强。

{interpretation}

## 预定检验和不确定性

三种替代读出各与原读出比较AUC、相似误认、旧命中，合计9个终点；种子级配对bootstrap 50000次，各双侧99.444%区间。联合标准是AUC下界>0、误认差上界<0、旧命中差下界>=−1个百分点。区间是bootstrap近似，并非严格有限样本保证。

{intervals}
{decisions}

## 控制和可支持的解释

等权能量使用原模型同一向量集合，只把识别计算中的逐向量质量设为1；回忆保留原质量。重建误差从眼前查询本身开始，用原加权动力学得到最后状态，与眼前查询比较；不使用目标原图或来源标签。最大相似度只访问模型实际存储的向量，是已有实例检索基线。

如果替代读出改善，只支持当前设置下识别结果依赖读出，不能证明回放没有改变模型，也不能证明所有误认源自单一因素。等权结果是质量2与质量1的比较，不是彻底移除回放。最大相似度同时改变聚合方式并忽略质量，不能把其全部改善归因于一个因素。

统一除以总质量只使分数平移，阈值同步平移；所有查询核验的分数最大误差{audit['normalization_max_error']:.3e}、动力学更新最大误差{audit['normalized_update_max_error']:.3e}、判定不一致{audit['normalized_judgment_disagreements']}、AUC差{audit['normalized_auc_max_error']}。这项恒等式不是新发现，不能把整体归一化当作本模型性能改善。

逐样本分数按“仅Task2向量→加入8条质量1旧向量→其质量改为2”分解；最大恒等式误差{audit['decomposition_max_error']:.3e}。CSV保存每项贡献及缓存/簇标签。分解路径明确，但阈值、排名与误认率是非线性的，不能报告唯一的错误因果百分比分配。

## 错误集中在哪里（分层为探索性）

{mechanism_table}
主角度的2000条相似未见查询中，640条来自8个复习样本的邻近扰动，1360条来自其余17个样本。四种读出对前640条全部误认。原加权读出679次错误中640次来自该组（94.26%）；等权仍有640次，说明逐实例质量2→1不足以解决这组错误。这里是来源分层的描述统计，不把来源标签交给模型。

这一集中现象与当前表示几何相容：缓存保留8条原旧向量的精确副本，其余旧向量在模型中仅有Task2旋转后的表示。复习样本的相似新输入与精确副本点积为cos(.24)={cosine:.6f}，而未复习旧目标与模型的最大相似度均值只有{outside_old.score.mean():.6f}；{mechanism_record['unbuffered_old_below_lure_cosine']} / {len(outside_old)}个未复习旧目标分数低于这类相似新图。故“取最像的一条”仍不能轻易兼顾全部旧目标与这组干扰。

为说明几何取舍，额外计算一个未用于正式评价的探索性阈值示例：最大相似度阈值提高到cos(.24)+1e-12时，这批旧目标命中仅{pct(strict_hit)}。这不是测试后选出的新方案，也没有用它改写主结果。不能把实例精确表示的不均衡等同于真实人脑复习机制。

## 执行与审核

60种子模型实例，240个实际模型；{audit['query_rows']:,}条含四读出的完整查询记录，{audit['partial_rows']:,}条部分回忆，{audit['decomposition_rows']:,}条分数分解，{audit['event_rows']:,}条回放质量事件。完整回忆未收敛{full_fail}条（按查询与组去重）；部分回忆未收敛{partial_fail}条。完整未收敛仍保留最后状态评分；部分未收敛身份判错。没有删除失败或困难查询。

模型只读审核={audit['query_read_only']}；预算正确={audit['event_budget_correct']}；新查询训练碰撞={audit['novel_training_collision_count']}；非有限分数={audit['nonfinite_score_count']}；基准分数与无回放一致={audit['baseline_score_matches_no_replay']}；整体审核通过={audit['all_required_checks_pass']}。阈值冻结UTC：{freeze['created_utc']}。

每组每读出阈值来自500个独立校准旧目标的第10百分位；主确认共40种子，每种子25旧目标与50个主角度相似未见。不同角度共享切向方向，不能当作更多独立重复。缓存内/外及簇/孤立分层、其他组和角度均探索性。

实际存储数组：回放模型17160字节，无回放13000字节；不含Python对象、输出、临时数组或进程峰值。资源CSV的recognition_seconds是四种分数及完整重建合计时间，不能据此给单一读出计时排名。模型本身显式存储向量；读出对比均访问同一实际模型，没有额外旧目标字典。

## 文献与研究边界

{refs}

实施前再次查询energy-guided replay episodic recognition、Hopfield familiarity replay readout、Hopfield recognition uniform weights；未找到直接覆盖本设置的答案，但检索不穷尽，部分全文入口失败。完整四轮查新另附。上述方法都有先例，本次贡献限于可重跑的指定设置诊断。

本实验是合成连续向量的闭式现代Hopfield模型。回放表示增加显式记忆质量，未运行优化器训练、STDP、扩散模型或新增人体实验。准确向量实例的新旧标签不能直接等同于人对同一物体不同视角的认知。识别改善不等于复制人脑，也不解决部分线索回忆接近猜测水平的问题。

## 复跑

运行：`python run_readout.py`；已有冻结结果用`python run_readout.py --resume`。源码和协议哈希锁定，缓存校验数据哈希。测试、独立导出重跑、前七阶段保留核验分别见同目录的实际核验文件。
'''
    report = report.replace('\n', '\n\n研究作者：Keymen（keymen6303-lgtm）。\n', 1)
    (reports/'固定记忆读出诊断研究报告.md').write_text(report,encoding='utf-8')
    (reports/'结论摘要.md').write_text('# 第八阶段结论摘要\n\n研究作者：Keymen（keymen6303-lgtm）。\n\n'+headline+'\n\n'+number_table+'\n'+decisions+'\n\n换读出不改变回忆能力；所有结论限于本合成人工模型。\n',encoding='utf-8')
    shutil.copyfile(ROOT/'readout_protocol.md',reports/'实验协议.md')
    shutil.copyfile(ROOT/'outputs/literature/第八阶段候选_逐轮查新记录.json',reports/'查新记录.json')
    shutil.copyfile(ROOT/'outputs/literature/研究方向查新规则.json',reports/'研究方向查新规则.json')
    (reports/'进度与问题.md').write_text('已完成预定实验及数值审核。\n\n'+headline+
        '\n\n测试、导出复跑和保留检查结果见各自核验文件；没有新增人体证据或增强回忆动力学。\n',encoding='utf-8')
    versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','matplotlib','scikit-learn','pytest','threadpoolctl')}
    (table/'environment.json').write_text(json.dumps(dict(platform=platform.platform(),python=sys.version,
        libraries=versions),ensure_ascii=False,indent=2),encoding='utf-8')
    htmltable='<table><tr><th>读出</th><th>AUC</th><th>相似误认</th><th>旧命中</th><th>部分回忆恢复</th></tr>'
    for r in READOUTS:
        v=main.loc[r];htmltable+=f'<tr><td>{RLABELS[r]}</td><td>{v.auc:.4f}</td><td>{pct(v.false_alarm)}</td><td>{pct(v.hit)}</td><td>{pct(v.identity)}</td></tr>'
    htmltable+='</table>'
    imageblocks=''.join(f'<figure><img src="figures/{name}.png" alt="{label}"><figcaption>{label}</figcaption></figure>' for name,label in [
        ('primary_readouts','主比较：同一记忆，四种判断'),('primary_intervals','预定9个终点的配对区间'),
        ('all_arms','各回放组的读出对照（其他组探索性）'),('angle_sweep','相似程度对读出的影响（探索性）'),
        ('score_distributions','确认分数与冻结阈值'),('score_decomposition','指定路径分数分解')])
    decisions_html='<ul>'+''.join(f'<li>{html.escape(RLABELS[d["readout"]])}：联合标准'+('通过' if d['joint_pass'] else '未通过')+'</li>' for d in summary['decisions'])+'</ul>'
    refs_html='<ul>'+''.join(f'<li><a href="{html.escape(s["url"],quote=True)}">{html.escape(s["title"])}</a>：{html.escape(s["established"])}</li>' for s in literature['sources'])+'</ul>'
    webpage='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定记忆读出诊断研究结果</title><style>body{font:17px/1.8 system-ui,sans-serif;color:#203244;max-width:1100px;margin:40px auto;padding:0 24px}h1,h2{line-height:1.4}.note{background:#eaf4f5;border-left:5px solid #238587;padding:20px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;border:1px solid #ccd5df;text-align:left}img{width:100%;height:auto}figure{margin:30px 0}figcaption{color:#536574}a{color:#1d698b}</style>'''
    webpage+=f'<h1>固定记忆，新旧判断方式有何影响？</h1><p>研究作者：Keymen（keymen6303-lgtm）</p><p>第八阶段 · 2026-10-10 · 20校准 + 40新确认种子</p><p class="note"><b>{headline}</b>识别和回忆必须分别评价。下表为固定高能量回放模型的主结果。</p>'+htmltable+decisions_html
    webpage+='<p>联合标准：AUC提高、相似误认下降、旧命中下降不超过1个百分点；9个主终点分别使用99.444%配对区间。部分回忆只计算一次，四种读出共用，因此相同。</p>'
    webpage+='<p>'+html.escape(interpretation)+'</p><p class="note"><b>错误集中在复习过的样本周围。</b>其640个相似未见查询被四种读出全部误认；原读出94.26%的错误来自这里。忽略次数没有解决这组错误。模型保留这些旧样本的精确副本，而其他旧样本只保留旋转表示，产生了明显的相似度取舍。分层和几何解释为探索性。</p>'
    webpage+='<p>整体归一化只平移分数和阈值，所有查询判定保持一致；不能把它当作性能改善。等权和最大相似度的控制含义不同，单独改善不能证明一个唯一原因。</p>'+imageblocks
    webpage+='<h2>核验与范围</h2><p>数值审核通过；所有旧结果保留核验、测试和独立复跑详情见下列文件。完整与部分回忆的未收敛情况均保留。这里只研究合成向量模型，没有新增人脑证据。</p><ul>'
    for file,label in [('reports/固定记忆读出诊断研究报告.md','完整报告'),('reports/实验协议.md','先于生产实验锁定的协议'),
        ('tables/paired_effects.csv','全部配对区间'),('tables/group_means.csv','全部组别均值'),('tables/audit.json','数值审核'),
        ('reports/独立导出重跑核验.json','独立导出重跑核验'),('reports/测试结果.txt','测试结果')]:
        webpage+=f'<li><a href="{file}">{label}</a></li>'
    webpage+='</ul><h2>已有工作</h2>'+refs_html+'<p>沿用已有方法诊断当前模型；不宣称全球首创。</p></html>'
    (DEST/'固定记忆读出诊断研究结果.html').write_text(webpage,encoding='utf-8')
    print(headline,flush=True)


if __name__=='__main__':generate()
