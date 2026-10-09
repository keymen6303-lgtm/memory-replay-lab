"""Chinese report and static scientific plots, generated from saved stage-9 results."""
from pathlib import Path
import json
import html
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
from .fidelity import ROOT,DEST,SOURCES,ARMS,READOUTS,CONFIG

SLABEL=dict(exact='精确旧副本',coherent='同一旋转副本',independent='等幅独立方向副本',none='无回放')
ALABEL=dict(high_energy='高能量优先',random_replay='随机回放',low_energy='低能量优先',no_replay='无回放')
RLABEL=dict(weighted='加权能量',max_similarity='最大相似度')
MLABEL=dict(auc='AUC',false_alarm='相似误认',hit='旧命中',identity='部分实例恢复',task2_hit='Task2完整命中')
COLORS=['#367c8b','#b77e37','#745e9f']
def pct(x):return f'{x*100:.2f}%'
def diff(x,metric):return f'{x:+.5f}' if metric=='auc' else f'{x*100:+.3f}个百分点'


def generate():
    font=Path('/System/Library/Fonts/STHeiti Medium.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,
        'axes.spines.right':False,'font.size':10})
    def save(name):
        plt.tight_layout();plt.savefig(DEST/'figures'/name,dpi=160);plt.close()
    tables=DEST/'tables';reports=DEST/'reports'
    summary=json.loads((tables/'summary.json').read_text())
    audit=json.loads((tables/'audit.json').read_text())
    diagnosis=json.loads((tables/'mechanism_diagnostic.json').read_text())
    freeze=json.loads((tables/'frozen_thresholds.json').read_text())
    group=pd.read_csv(tables/'group_means.csv',float_precision='round_trip')
    effects=pd.read_csv(tables/'paired_effects.csv',float_precision='round_trip')
    subgroup=pd.read_csv(tables/'subgroup_means.csv')
    q=pd.read_csv(tables/'queries.csv',float_precision='round_trip')
    q=q[q.cohort=='confirmation']
    geometry=pd.read_csv(tables/'geometry.csv');geometry=geometry[geometry.cohort=='confirmation']
    resources=pd.read_csv(tables/'resources.csv')
    main=group[(group.arm=='high_energy')&(group.readout=='weighted')&(group.angle==.24)].set_index('source').loc[list(SOURCES)]
    base=group[(group.arm=='no_replay')&(group.readout=='weighted')&(group.angle==.24)].iloc[0]
    diag={ (x['source'],x['arm'],x['readout']):x for x in diagnosis['counts'] }
    direct=effects[effects.primary & (effects.comparison=='source_minus_exact')].set_index('metric')
    joint=summary['decisions']['joint_pass']
    headline='等幅独立方向副本通过预定联合改善标准。' if joint else '取消精确副本条件未降低主读出的相似误认，预定联合改善标准未通过。'
    # Plot means, explicitly avoiding query-level uncertainty.
    fig,axes=plt.subplots(1,4,figsize=(13,4.5))
    for ax,metric in zip(axes,['auc','false_alarm','hit','identity']):
        scale=1 if metric=='auc' else 100
        values=main[metric]*scale
        ax.bar(np.arange(3),values,color=COLORS)
        ax.set_xticks(np.arange(3),[SLABEL[s] for s in SOURCES],rotation=25,ha='right')
        ax.axhline(base[metric]*scale,ls=':',color='#666',label='无回放参考（预算较低）')
        ax.set_title(MLABEL[metric]);ax.set_ylabel('AUC' if metric=='auc' else '%')
        ax.set_ylim(0,1 if metric=='auc' else 100 if metric in ('false_alarm','hit') else 6)
        ax.legend(fontsize=7)
    fig.suptitle('高能量回放 · 加权能量主读出 · θ=.24 · 40个确认种子均值')
    save('primary_source_controls.png')
    fig,axes=plt.subplots(1,5,figsize=(14,3.8))
    for ax,metric in zip(axes,['auc','false_alarm','hit','identity','task2_hit']):
        row=direct.loc[metric];scale=1 if metric=='auc' else 100
        ax.errorbar([row['mean']*scale],[0],xerr=[[max(0.,(row['mean']-row.low)*scale)],
            [max(0.,(row.high-row['mean'])*scale)]],fmt='o',capsize=4,color='#367c8b')
        ax.axvline(0,color='#999',ls='--');ax.set_yticks([]);ax.set_title(MLABEL[metric])
        if metric in ('hit','identity','task2_hit'):
            margin=dict(hit=1,identity=.25,task2_hit=1)[metric]
            ax.axvline(-margin,color='#b77e37',ls=':',label=f'−{margin:g}pp保护界限');ax.legend(fontsize=7)
        ax.set_xlabel('差值' if metric=='auc' else '差值（百分点）')
    fig.suptitle('等幅独立方向 − 精确源：14个主终点各99.643%配对区间')
    save('primary_intervals.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    policy=effects[effects.primary&(effects.comparison=='high_minus_random')]
    for ax,metric in zip(axes,['auc','false_alarm','identity']):
        d=policy[policy.metric==metric].set_index('source').loc[['exact','independent']]
        scale=1 if metric=='auc' else 100
        ax.errorbar(d['mean']*scale,np.arange(2),xerr=np.array([
            (d['mean']-d.low)*scale,(d.high-d['mean'])*scale]),fmt='o',capsize=4)
        ax.axvline(0,color='#999',ls='--');ax.set_yticks(np.arange(2),[SLABEL[s] for s in d.index])
        ax.set_title(MLABEL[metric]);ax.set_xlabel('高能量 − 随机' if metric=='auc' else '高能量 − 随机（百分点）')
    fig.suptitle('选择策略效应：精确源与变形源各自比较 · 主区间99.643%')
    save('selection_effects.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric in zip(axes,['false_alarm','hit']):
        matrix=np.array([[group[(group.source==s)&(group.arm=='high_energy')&(group.readout==r)&(group.angle==.24)][metric].iloc[0]*100 for r in READOUTS] for s in SOURCES])
        im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=100)
        for i in range(3):
            for j in range(2):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>60 else '#203040')
        ax.set_xticks([0,1],[RLABEL[r] for r in READOUTS]);ax.set_yticks(range(3),[SLABEL[s] for s in SOURCES])
        ax.set_title(MLABEL[metric]+'（%）');fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('源条件与读出的组合：最大相似度比较为探索性')
    save('source_readout_diagnostic.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,r in zip(axes,READOUTS):
        buffered=[diag[(s,'high_energy',r)]['buffered_false_alarm']*100 for s in SOURCES]
        outside=[diag[(s,'high_energy',r)]['outside_false_alarm']*100 for s in SOURCES]
        pos=np.arange(3)
        ax.bar(pos-.18,buffered,width=.36,label='回放内父样本附近：640条',color='#367c8b')
        ax.bar(pos+.18,outside,width=.36,label='回放外父样本附近：1360条',color='#b77e37')
        ax.set_xticks(pos,[SLABEL[s] for s in SOURCES],rotation=23,ha='right')
        ax.set_ylim(0,105);ax.set_ylabel('相似误认率（%）');ax.set_title(RLABEL[r]);ax.legend(fontsize=7)
    fig.suptitle('分层只用于评估；未作为判断器输入')
    save('buffered_lures.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,s in zip(axes,['exact','independent']):
        g=q[(q.source==s)&(q.arm=='high_energy')&(q.readout=='weighted')]
        for buffered,color in [(True,'#367c8b'),(False,'#b77e37')]:
            lure=g[(g.kind=='lure')&(g.theta==.24)&(g.parent_buffered==buffered)]
            ax.hist(lure.score,bins=30,alpha=.6,density=True,color=color,label='回放内相似新' if buffered else '回放外相似新')
        t=freeze['thresholds'][s+'__high_energy']['weighted']
        ax.axvline(t,color='#333',ls='--',label='冻结阈值')
        ax.set_title(SLABEL[s]);ax.set_xlabel('加权能量熟悉分数');ax.set_ylabel('密度');ax.legend(fontsize=8)
    fig.suptitle('主读出：源变形后回放内相似输入仍跨过旧阈值')
    save('lure_score_distributions.png')
    main_table='|高能量回放源|AUC|相似误认|旧命中|部分实例恢复|Task2完整命中|\n|---|---:|---:|---:|---:|---:|\n'
    for s in SOURCES:
        v=main.loc[s];main_table+=f'|{SLABEL[s]}|{v.auc:.5f}|{pct(v.false_alarm)}|{pct(v.hit)}|{pct(v.identity)}|{pct(v.task2_hit)}|\n'
    main_table+=f'|无回放参考（预算较低）|{base.auc:.5f}|{pct(base.false_alarm)}|{pct(base.hit)}|{pct(base.identity)}|{pct(base.task2_hit)}|\n'
    interval_table='|预定比较|指标|差值|99.643%区间|\n|---|---|---:|---|\n'
    comparison_labels=dict(source_minus_exact='高能量：独立方向源−精确源',high_minus_random='高能量−随机',policy_interaction_independent_minus_exact='选择策略效应：独立方向−精确源')
    for row in summary['primary_effects']:
        label=comparison_labels[row['comparison']]
        if row['comparison']=='high_minus_random':label+='（'+SLABEL[row['source']]+'）'
        metric=row['metric'];interval_table+=f'|{label}|{MLABEL[metric]}|{diff(row["mean"],metric)}|[{diff(row["low"],metric)}, {diff(row["high"],metric)}]|\n'
    diagnostic_table='|高能量源|读出|相似误认|旧命中|回放内相似新错误数/640|\n|---|---|---:|---:|---:|\n'
    for s in SOURCES:
        for r in READOUTS:
            v=group[(group.source==s)&(group.arm=='high_energy')&(group.readout==r)&(group.angle==.24)].iloc[0]
            diagnostic_table+=f'|{SLABEL[s]}|{RLABEL[r]}|{pct(v.false_alarm)}|{pct(v.hit)}|{diag[(s,"high_energy",r)]["buffered_wrong"]}/640|\n'
    limits='''本结论仅限N24、c=.35、beta8、eps=.05、theta=.24的合成显式向量/质量模型。回放是增加质量，不是STDP或优化器学习。本实验没有新增人类放电数据，也没有人体突触观测。独立方向控制同时改变源间相关性；同一旋转控制减少独立支持点数量。高能量的优势可能依赖单一孤立样本与簇结构，不代表所有经历普遍改善。主要区间为种子级近似bootstrap；探索性95%区间没有校正全部探索比较，不能用其代替主检验。Task2指标只测试完整输入是否认作曾学，没有测试其部分线索恢复。'''
    decisions='\n'.join('- '+k+'：'+str(v) for k,v in summary['decisions'].items())
    max_effect=effects[(effects.comparison=='source_minus_exact')&(effects.source=='independent')&(effects.readout=='max_similarity')&(effects.angle==.24)].set_index('metric')
    sg=subgroup[(subgroup.arm=='high_energy')&(subgroup.readout=='weighted')&(subgroup.field=='parent_group')]
    sg_text='；'.join(f'{SLABEL[v.source]}：{v.value}恢复{pct(v.identity)}' for v in sg.itertuples())
    replay=resources[resources.arm!='no_replay'];none=resources[resources.arm=='no_replay']
    geo=geometry[geometry.source=='independent'];angles=np.arccos(np.clip(geo.replay_cosine,-1,1))
    text=f'''# 第九阶段：回放源精度与相似误认研究报告

{headline}

## 实际执行

协议在生产前锁定；20新校准种子20273001–20273020，40新确认种子20274001–20274040，编号不是日期。每seed共10个实际模型（3源×3选择策略及无回放参考），2个读出，查询与部分线索完全共享。阈值由校准500个旧目标分数第10百分位决定，冻结后才生成确认样本。共{audit['rows']['queries']}条读出查询记录（含2读出，对应135000个查询—模型对）、{audit['rows']['partial']}条部分线索恢复记录、{audit['rows']['events']}次回放质量事件、60个原始NPZ。

## 主结果

{main_table}

高能量组等幅独立方向源减精确源的误认差为{diff(direct.loc['false_alarm','mean'],'false_alarm')}，99.643%区间[{diff(direct.loc['false_alarm','low'],'false_alarm')}, {diff(direct.loc['false_alarm','high'],'false_alarm')}]；区间跨0，不能称改善，也不能称已经证明两者等效。AUC改善亦未获支持。旧命中、部分恢复和Task2完整命中的预定保护通过，但联合改善失败。

{decisions}

## 14个预定主区间

{interval_table}

精确源和独立方向源中，高能量相对随机的AUC及部分恢复差区间均在0上方，误认差区间在0下方，说明这些条件下的小幅选择收益有证据。三项源×选择交互区间均跨0，不能宣称源条件完全不影响策略效果或已证明等效。整体部分恢复仍约4.5%，接近25个候选均匀猜测的1/25=4%参照；这个参照不是评估器实际随机过程。

## 错误集中与读出诊断

{diagnostic_table}

三个源条件的加权能量判断，对回放内640条主角度相似新输入仍全部误认。精确源和独立方向源均为690/2000误认，其中640/690={pct(640/690)}来自回放内。故精确副本并非当前主读出产生这一错误集中的必要条件，上一阶段的精度不均衡解释不足。

最大相似度下，独立方向源相对精确源的误认差{diff(max_effect.loc['false_alarm','mean'],'false_alarm')}，探索性95%区间[{diff(max_effect.loc['false_alarm','low'],'false_alarm')}, {diff(max_effect.loc['false_alarm','high'],'false_alarm')}]；但旧命中差{diff(max_effect.loc['hit','mean'],'hit')}，区间[{diff(max_effect.loc['hit','low'],'hit')}, {diff(max_effect.loc['hit','high'],'hit')}]。这显示源与读出的组合会显著改变表现，并伴有旧命中代价；非预定主成功结果，不能据此宣布完成修复。

同一旋转副本仅重复已有Y支持点。最大相似度分数、阈值与判断均与无回放相同，这是实现核验的恒等式，不是新发现。加权能量下重复项仍有作用：把8个Y的质量从1变成3，与保留33个条目的模型能量/更新等价（最大误差{audit['coherent_collapsed_max_error']:.3g}）。因此它的35.85%误认不能用“仍有精确X副本”解释。它提示回放造成的相对质量和局部聚集仍需考虑，但本阶段并未把所有原因唯一分解，也没有提出未经查新的新算法。

探索性簇/孤立恢复：{sg_text}。回放外旧内容部分恢复全部为0，模型没有实现全面内容恢复。

## 几何、预算与审核

等幅独立方向回放的角误差均值{float(angles.mean()):.5f}rad（{float(np.degrees(angles.mean())):.2f}°），逐项匹配基础旋转内积的最大误差{audit['independent_cosine_max_error']:.3g}；全模型单位范数最大误差{audit['unit_norm_max_error']:.3g}。不同策略对同一内容使用同一源方向，不依据查询方向挑选。实际源向量、完整Gram诊断统计和源角度在NPZ及geometry/resources表中。

回放模型实际Memory向量/质量数组{int(replay.memory_array_bytes.iloc[0])}B，无回放{int(none.memory_array_bytes.iloc[0])}B。只计这两个持久数组，不含源生成器全字典、编号、日志、临时运算、Python对象或峰值内存，不作为真实脑资源量。计时列同时包含两读出和部分回忆，不能称单一算法性能基准。

exact逐数组复用原模型核验通过；跨源编号与事件相同，64事件、8条、总质量16正确，回放组数组字节数相同。新输入与实际训练模型碰撞数{audit['novel_training_collisions']}，非有限分数{audit['nonfinite_scores']}，部分回忆未收敛{audit['partial_nonconvergence_count']}。没有删掉失败样本。coherent最大相似度数值误差{audit['coherent_max_similarity_error']}，判定不一致{audit['coherent_max_judgment_disagreements']}。审计all_required_checks_pass={audit['all_required_checks_pass']}。

## 已有研究和新颖性边界

已完成五轮查新，8项原始来源及实际读取范围保存在查新记录。本次复用Takeda等2026预印本闭式模型，沿用已有带噪回放和漂移问题做对照。McCallum2007的能量比筛选不是本项目绝对能量排序；类增量漂移补偿不是固定编码器的合成旋转。没有查到直接回答当前窄条件的结果，仍不能声称无人做过或全球首次。部分来源仅可读取摘要/原文搜索片段，完整边界见查新记录。

## 结论与限制

主结果修正了上一阶段解释：去掉精确旧副本条件，没有消除加权能量的相似误认；高能量相对随机的小收益在两种源条件下分别有证据。来源精度对另一种读出仍很重要，不能泛化为精度完全无关。当前模型整体记忆仍未成功，恢复很低且回放后的新旧区分弱于较低预算的无回放参考。

{limits}

入口：python run_fidelity.py；已有冻结结果用--resume。环境与独立重跑、缓存继续、原有输出保留、测试及项目包核验文件另存reports。独立导出使用本机既有锁定依赖，不声称跨系统或重新安装环境通过。
'''
    verification=reports/'独立导出重跑核验.json'
    if verification.exists():
        v=json.loads(verification.read_text())
        tests=(reports/'测试结果.txt').read_text().strip().splitlines()[-1]
        export_tests=(reports/'独立导出测试.txt').read_text().strip().splitlines()[-1]
        continuation=json.loads((reports/'断点继续核验.json').read_text())
        preservation=json.loads((reports/'前八阶段保留核验.json').read_text())
        text+=f'''\n## 已完成的工程核验

全项目测试：{tests}；独立包测试：{export_tests}。独立项目从零重算{len(v['exact_tables'])}张CSV、{v['raw_npz_files']}个NPZ中的{v['exact_raw_arrays']}个数组及冻结阈值完全一致；资源表除运行时长外相同。全部已保存判定和校准阈值由分数重建通过，确认案例创建时间晚于阈值冻结。断点继续{continuation['resume_cache_hits']}/60例命中，冻结哈希不变；{preservation['checked_existing_output_files']}个既有outputs文件哈希未变。核验仅为当前机器既有环境。查新记录是执行前的候选快照，其中“尚未执行”描述的是当时状态；本报告及运行记录给出当前已执行结果。\n'''
    (reports/'回放源精度研究报告.md').write_text(text,encoding='utf-8')
    (reports/'结论摘要.md').write_text('# 第九阶段结论\n\n'+headline+'\n\n'+main_table+'\n高能量相对随机的小收益在两种源中分别得到支持，但整体恢复仍很低。最大相似度与变形源的组合为探索性结果，误认降低伴有旧命中代价。仅限合成人工模型，没有新增人类机制证据。\n',encoding='utf-8')
    shutil.copyfile(ROOT/'fidelity_protocol.md',reports/'实验协议.md')
    for name,target in [('第九阶段候选_逐轮查新记录.json','查新记录.json'),('研究方向查新规则.json','研究方向查新规则.json')]:
        shutil.copyfile(ROOT/'outputs/literature'/name,reports/target)
    environment=dict(python=sys.version,platform=platform.platform(),dependencies={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib','scikit-learn','pytest','threadpoolctl']})
    (tables/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    figures=[('primary_source_controls.png','主结果：三个源条件'),('primary_intervals.png','源变化的五项主终点区间'),('selection_effects.png','高能量相对随机的效果'),('source_readout_diagnostic.png','源与读出的探索性组合'),('buffered_lures.png','错误是否仍集中于回放内容附近'),('lure_score_distributions.png','变形后分数是否仍跨过阈值')]
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第九阶段：回放源精度研究结果</title><style>body{font:17px/1.8 system-ui,sans-serif;max-width:1120px;margin:35px auto;padding:0 24px;color:#243649}h1,h2{line-height:1.4}img{width:100%;height:auto;margin:15px 0}pre{white-space:pre-wrap;background:#f4f7fa;padding:18px;font-size:14px}.note{background:#eef5fa;border-left:5px solid #357099;padding:20px}a{color:#1b6199}li{margin:8px 0}</style><h1>回放副本变形后：主读出的误认仍然存在</h1><p>第九阶段 · 20个校准种子 + 40个新确认种子 · 真实运行结果</p>'''
    page+='<p class="note"><b>'+html.escape(headline)+'</b><br>精确与等幅独立方向源的误认均34.50%；来源变化并未让原判断方式获得联合改善。高能量选择仍有小幅收益，但部分恢复仅约4.5%。本结果修正了上一轮解释，不能证明人脑机制。</p>'
    page+='<h2>主要数字</h2><pre>'+html.escape(main_table)+'</pre><h2>关键解释</h2><p>去掉精确副本没有消除加权能量下回放内640条相似新输入的误认。最大相似度下误认明显减少，但旧命中也下降，且该比较为探索性。不能把主失败换成探索成功。相同旋转副本减少独立支持点数量，最大相似度与无回放完全相同属于已知恒等式。</p>'
    for name,title in figures:page+='<h2>'+html.escape(title)+'</h2><img src="figures/'+name+'" alt="'+html.escape(title)+'">'
    page+='<h2>完整报告和可核查记录</h2><ul>'+''.join('<li><a href="'+path+'">'+label+'</a></li>' for path,label in [('reports/回放源精度研究报告.md','完整中文报告'),('reports/实验协议.md','运行前协议'),('reports/查新记录.json','五轮查新及读取边界'),('tables/summary.json','14个主区间与决定'),('tables/audit.json','工程及数据审核'),('tables/frozen_thresholds.json','校准冻结阈值'),('tables/mechanism_diagnostic.json','错误分层计数'),('tables/group_means.csv','全条件均值表')])+'</ul><p>'+html.escape(limits)+'</p></html>'
    if verification.exists():
        extra='<h2>独立复现与交付核验</h2><p>'+html.escape(tests)+'；独立导出原始数组、表格与阈值核验通过。</p><ul>'
        for filename,label in [('独立导出重跑核验.json','从零重跑核验'),('断点继续核验.json','缓存与冻结核验'),('前八阶段保留核验.json','既有输出保留核验'),('测试结果.txt','全项目测试日志')]:
            extra+='<li><a href="reports/'+filename+'">'+label+'</a></li>'
        page=page.replace('</html>',extra+'</ul></html>')
    (DEST/'回放源精度研究结果.html').write_text(page,encoding='utf-8')
    print('中文报告与6张科学图已生成',flush=True)
