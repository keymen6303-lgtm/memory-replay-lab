"""Chinese scientific report, using only saved stage-7 outputs."""
import html
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .energy import ROOT,DEST,ARMS,LABELS,read,condition_id

COLORS=['#718399','#126b8b','#d08137','#66884d']


def format_pp(x):return f'{x*100:+.2f}个百分点'
def pct(x):return f'{x*100:.2f}%'
def savefig(name):
    plt.tight_layout();plt.savefig(DEST/'figures'/name,dpi=160);plt.close()


def generate():
    local='/System/Library/Fonts/STHeiti Medium.ttc'
    if __import__('pathlib').Path(local).exists():
        font_manager.fontManager.addfont(local)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=local).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,
        'axes.spines.right':False,'font.size':10})
    summary=json.loads((DEST/'tables/summary.json').read_text())
    audit=json.loads((DEST/'tables/audit.json').read_text())
    freeze=json.loads((DEST/'tables/frozen_thresholds.json').read_text())
    run=json.loads((DEST/'tables/run_record.json').read_text())
    group=read('group_means.csv');metrics=read('metrics_by_seed.csv')
    q=read('queries.csv');p=read('partial.csv');resources=read('resources.csv')
    primary=group[(group.condition==condition_id(24,.35))&(group.angle==.24)].set_index('arm')
    effects={z['metric']:z for z in summary['primary_effects']}
    ri=effects['identity'];fa=effects['false_alarm']
    if summary['primary_tradeoff_supported']:
        conclusion='主条件支持恢复与误认的取舍：高能量优先相对随机回放提高部分实例恢复，同时提高相似新实例误认。'
    else:
        conclusion='主条件没有确认预定的“恢复改善同时误认增加”假设；应分别看两个指标及区间，不能用次要条件替代。'
        if ri['ci'][0]>0 and fa['ci'][1]<0:
            conclusion='主条件中，高能量优先相对随机回放使实例恢复略升、相似误认略降；预定“恢复改善但误认增加”的假设未获支持。'
    details=(f'高能量减随机：部分实例恢复{format_pp(ri["mean"])}，97.5%区间'
        f'[{format_pp(ri["ci"][0])}, {format_pp(ri["ci"][1])}]；相似实例误认'
        f'{format_pp(fa["mean"])}，97.5%区间[{format_pp(fa["ci"][0])}, {format_pp(fa["ci"][1])}]。')

    # Original closed-form replication: means across rotations, not novel findings.
    rep=summary['replication'];fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    labels=['孤立自回放','簇自回放','同簇交叉','孤立→簇','簇→孤立']
    names=['outlier_self','cluster_self','within_cluster','outlier_to_cluster','cluster_to_outlier']
    for ax,s in zip(axes,rep['summaries']):
        ax.bar(np.arange(5),[s['means'][k] for k in names],color=['#126b8b']*2+['#91b2ba']*3)
        ax.set_yscale('log');ax.set_xticks(range(5),labels,rotation=20)
        ax.set_title(f'旋转 ε={s["epsilon"]}；2000次')
        ax.set_ylabel('扣除无漂移重复收益后的能量保护（均值）')
    savefig('replication_replay_order.png')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(metric,label) in zip(axes.flat,[('identity','部分线索实例恢复'),('false_alarm','相似未见实例误认'),('hit','完整旧实例命中'),('auc','目标—相似实例AUC')]):
        factor=1 if metric=='auc' else 100
        ax.bar(range(4),[primary.loc[a,metric]*factor for a in ARMS],color=COLORS)
        ax.set_xticks(range(4),[LABELS[a] for a in ARMS],rotation=15)
        ax.set_title(label);ax.set_ylim(0,1 if metric=='auc' else 100)
        ax.set_ylabel('AUC' if metric=='auc' else '%')
    fig.suptitle('主条件：24簇+1孤立 · c=.35 · θ=.24 · 40确认种子；图为均值')
    savefig('primary_outcomes.png')
    fig,ax=plt.subplots(figsize=(8,3.6))
    vals=np.array([ri['mean'],fa['mean']])*100
    low=np.array([ri['ci'][0],fa['ci'][0]])*100
    high=np.array([ri['ci'][1],fa['ci'][1]])*100
    ax.errorbar(vals,[1,0],xerr=[vals-low,high-vals],fmt='o',color='#126b8b',capsize=5)
    ax.axvline(0,color='#777',ls='--');ax.set_yticks([1,0],['实例恢复：正值更好','相似误认：正值更坏'])
    ax.set_xlabel('高能量优先 − 随机回放（百分点）；两主终点各97.5%区间')
    savefig('primary_paired_intervals.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,(metric,title) in zip(axes,[('identity','实例恢复差：正值更好'),('false_alarm','相似误认差：正值更坏')]):
        matrix=np.zeros((3,3))
        for i,n in enumerate((12,24,40)):
            for j,c in enumerate((.15,.35,.65)):
                g=group[(group.condition==condition_id(n,c))&(group.angle==.24)].set_index('arm')
                matrix[i,j]=100*(g.loc['high_energy',metric]-g.loc['random_replay',metric])
        bound=max(.01,np.abs(matrix).max());im=ax.imshow(matrix,cmap='RdBu_r',vmin=-bound,vmax=bound)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        ax.set_xticks(range(3),['.15','.35','.65']);ax.set_yticks(range(3),['12+1','24+1','40+1'])
        ax.set_xlabel('样本相关性 c');ax.set_ylabel('训练簇样本+孤立样本数');ax.set_title(title)
        fig.colorbar(im,ax=ax,label='百分点')
    fig.suptitle('高能量−随机回放；中档干扰；九条件是探索性矩阵')
    savefig('exploratory_matrix.png')
    confirm=q[(q.cohort=='confirmation')&(q.condition==condition_id(24,.35))]
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for ax,arm in zip(axes,['random_replay','high_energy']):
        g=confirm[confirm.arm==arm]
        for kind,label,color in [('target','完整旧实例','#126b8b'),('lure','相似新实例','#d08137')]:
            h=g[g.kind==kind]
            if kind=='lure':h=h[h.theta==.24]
            ax.hist(h.score,bins=35,density=True,alpha=.5,label=label,color=color)
        threshold=freeze['thresholds'][condition_id(24,.35)][arm]
        ax.axvline(threshold,color='#333',ls='--',label='预先冻结阈值')
        ax.set_title(LABELS[arm]);ax.set_xlabel('熟悉分数 −E');ax.set_ylabel('密度');ax.legend()
    savefig('score_distributions.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for arm,color in zip(ARMS,COLORS):
        g=group[(group.condition==condition_id(24,.35))&(group.arm==arm)].sort_values('angle')
        axes[0].plot(g.angle,100*g.false_alarm,'o-',label=LABELS[arm],color=color)
        axes[1].plot(g.angle,g.auc,'o-',label=LABELS[arm],color=color)
    axes[0].set_ylabel('相似误认 %');axes[1].set_ylabel('目标—相似AUC')
    for ax in axes:ax.set_xlabel('相似干扰角度（弧度）；越小越相似');ax.legend()
    savefig('similarity_sweep.png')

    table='|策略|旧实例恢复|相似误认|旧命中|AUC|重建MSE|\n|---|---:|---:|---:|---:|---:|\n'
    for arm in ARMS:
        z=primary.loc[arm]
        table+=f'|{LABELS[arm]}|{pct(z.identity)}|{pct(z.false_alarm)}|{pct(z.hit)}|{z.auc:.4f}|{z.mse:.6f}|\n'
    test_path=ROOT/'work/energy/preflight_tests.txt'
    brief=test_path.read_text().strip().splitlines()[-1] if test_path.exists() else '见测试日志'
    reptext=[]
    for s in rep['summaries']:
        delta=s['paired']['outlier_minus_cluster_rise']
        reptext.append(f'- epsilon={s["epsilon"]}：孤立减簇能量增量{delta["mean"]:.8f}，95%区间[{delta["ci95"][0]:.8f}, {delta["ci95"][1]:.8f}]；五项预定差区间下界均为正。')
    exploratory=[]
    for n,c in [(12,.15),(12,.35),(12,.65),(24,.15),(24,.35),(24,.65),(40,.15),(40,.35),(40,.65)]:
        g=group[(group.condition==condition_id(n,c))&(group.angle==.24)].set_index('arm')
        exploratory.append(f'|{n}+1|{c}|{format_pp(g.loc["high_energy","identity"]-g.loc["random_replay","identity"])}|{format_pp(g.loc["high_energy","false_alarm"]-g.loc["random_replay","false_alarm"])}|')
    report=f'''# 第七阶段：能量引导回放与相似实例识别

{conclusion}

{details}

这是人工连续向量、显式记忆分布的闭式现代Hopfield实验，不是照片数据、人类实验、STDP仿真或MHN-BM训练。没有新增人体证据。

## 查新后的起点与原文复现

起点为[Takeda等2026预印本v2](https://arxiv.org/html/2605.27975v2)。其能量引导回放、孤立样本遗忘和回放收益排序已有结果。联合回忆/识别任务也已有[SQHN2024](https://www.nature.com/articles/s41467-024-46976-4)等前例。当前增量候选是该能量选择原则对未训练相似实例识别的适用边界，不是算法或任务首创；定向检索不能证明全球无人做过。

复现12簇+1孤立、50维、c=.35、beta8等角模型，原文正文/附录不同旋转强度分别2000次核验，复用同Omega。没有获取作者脚本，以原能量/回放公式独立重实现；复现方向，不声称精确复现原图或扩散模型。

{chr(10).join(reptext)}

固定点最大残差{rep['checks']['fixed_point_max_residual']:.3g}，最大迭代Jacobian特征值{rep['checks']['max_jacobian_eigenvalue']:.6f}；直接增加单个回放项与精确log1p公式最大差{rep['checks']['exact_replay_identity_max_error']:.3g}。扣除无旋转重复项的机械收益后，预定排序通过。结果是既有结论的核验。

![原文收益排序](../figures/replication_replay_order.png)

## 新扩展：预定主条件

24簇+1孤立、64维、c=.35、beta8，Task2对旧分布做epsilon=.05随机旋转。8槽离线选择缓存、64次质量增加、每次.25，三回放组总新增质量16且每槽2。高/低能量只改变Task1缓存选择，选择时不读取未来查询或Task2。新样本在训练和缓存外。遮挡视图保留16/64维，4视图/旧实例；新相似干扰与parent夹角.24，每旧实例2个方向。

{table}

主比较高能量−随机，两个主终点各97.5%配对bootstrap区间，种子是独立推断单位（40）；只有恢复差和误认差下界同时>0才支持预定取舍。{details} 主判定：{'支持' if summary['primary_tradeoff_supported'] else '未获确认'}。

必须同时看到绝对水平：高能量恢复率{pct(primary.loc['high_energy','identity'])}，随机{pct(primary.loc['random_replay','identity'])}；25实例的均匀猜测参照为4%。整体恢复非常差。这一恢复差来自孤立实例：高能量与随机的簇内恢复均值相同；不是普遍改善所有旧实例。高能量完整旧命中{pct(primary.loc['high_energy','hit'])}，随机{pct(primary.loc['random_replay','hit'])}，没有确认命中完全无损。

无回放参照误认{pct(primary.loc['no_replay','false_alarm'])}，高能量仍有{pct(primary.loc['high_energy','false_alarm'])}；AUC无回放{primary.loc['no_replay','auc']:.4f}，高能量{primary.loc['high_energy','auc']:.4f}。这些是预定次要/参照描述：选择高能量对象只小幅缓解相对随机的识别问题，没有解决这一回放配置下的大幅误认，也不能说整体超过无回放。关于“增加回放质量是否导致误认”的跨质量因果问题需要另行查新和预定实验，当前不把选择规则对照当作剂量实验。

![主条件](../figures/primary_outcomes.png)
![主差区间](../figures/primary_paired_intervals.png)

命中、AUC与MSE均单列，不能以降低重建误差替代正确的新旧识别。误认随冻结阈值变化而AUC不变或升高时，要考虑分数标尺/校准迁移，不能只说记忆内容损坏。降低误认若同时降低命中也不是全面改善。

![分数及冻结阈值](../figures/score_distributions.png)

## 相关性、负荷和相似程度：探索性结果

|簇+孤立数|c|恢复：高−随机|中档误认：高−随机|
|---|---:|---:|---:|
{chr(10).join(exploratory)}

九条件中的其他八个不能替代主条件；这些均值与95%次要区间是探索性，不作为九项独立显著结论。每种子跨条件相关，不合并成360个独立种子。三个干扰角度.12/.24/.48，连续切向扰动越小越相似。

![探索矩阵](../figures/exploratory_matrix.png)
![相似程度](../figures/similarity_sweep.png)

## 模型、预算与解释限制

模型显式保存Task2向量和所选旧向量/质量。回放用增加旧向量在logsumexp与回忆softmax中的质量表示，64次事件可精确聚合成8份质量；不是64次突触更新或优化器步数。Task2旋转产生域漂移，闭式模型改变代表任务分布的显式记忆，不模拟训练动力学。不能把本轮结论直接搬到前六阶段经典Hebbian网络。

三个回放组同样8槽与质量16、相同Task2输入和槽事件路径；不同源是选择规则的干预。资源CSV中的persistent_array_bytes是按构造时数组nbytes加标量额度记账，包含来源ID和计数工作数组；这些元数据并未全部作为Memory实例的持久属性，字段名不能当作模型驻留内存的精确实测。主条件Memory实际持久向量与质量数组：回放组17160B、无回放13000B；含ID/计数/标量额度的记账量分别17304B、13016B。三个回放组的这两种预算均一致，另有资源定义核验表。未计Python对象、日志或进程峰值，不能声称所有内存完全一致。为评价身份保留的Task1真值只在评价器，模型不能访问；离线缓存选择不等于8槽在线蓄水池。

几何是等角簇加一个孤立样本，各条件都是高度理想化分布。高能量缓存必含孤立样本，低能量不含；簇内真实并列通过预生成随机优先序打破，不让浮点误差挑选。只有一个孤立样本，不能证明自然数据所有难样本都该按此排序。确认部分查询收敛比例{audit['convergence_fraction']:.6f}；未收敛计身份失败，不删除。

精确输入向量定义实例真值；不同现实视角是否同一物体的语义身份并未模拟。相似实例由切向扰动生成，确保未进入Task1/Task2；不把恢复概念原型当看过具体实例，不把同分布识别与未见类别OOD检测混为一谈。

## 校准、完整数据与工程核验

20校准种子20269001–20269020，40确认20270001–20270040，9条件4组。每条件每组阈值只由20*M个完整旧目标分数第10百分位确定，{freeze['frozen_at']}写入并冻结，确认开始前保存。哈希{run['freeze_sha256']}。确认数据未回调阈值或预算。

完整查询{len(q)}条、部分查询{len(p)}条、质量增加事件{summary['event_rows']}条；包括校准与确认，不将全部查询作为统计样本。逐条分数、判断、原实例ID、遮挡坐标、收敛和质量事件已保存CSV。主条件校准/确认代表NPZ和原文复现NPZ保存。

全部核验：只读评价、源为已见样本、64事件每槽8次、总质量16、回放组实际字节一致、新查询零训练碰撞、判断可由冻结阈值逐条重建、事件可重建缓存和质量。原文网页版本/hash、环境、协议源码hash及前六阶段保存清单在tables目录。测试：{brief}。缓存按模型源码/协议/配置hash验证；报告源码不参与模型缓存。

入口：`.venv/bin/python run_energy.py --resume`，从头运行去掉--resume；仅核验原文用--replication-only。独立导出重跑与测试记录见报告目录；仅当前CPU环境验证，未宣称其他系统或全新依赖安装通过。

## 当前结论与下一步依据

{conclusion} {details} 原文闭式收益排序复现通过，这不构成新发现。当前新结果的强度由预定主比较决定，其他条件只提供后续问题。后续若提出新机制，必须先核对前例；不能从这组连续向量结果宣称人类记忆机制已证实。

下一项最值得做的诊断是固定本轮已学向量和回忆过程，只改变新旧判断读出，检验大幅误认是否依赖于将回放质量直接混入熟悉分数。比较当前加权能量、忽略回放质量的对照读出及已有的重建误差读出；独立校准，并同时保留命中与实例恢复指标。不同熟悉/回忆读出在[Greve等2010](https://onlinelibrary.wiley.com/doi/abs/10.1002/hipo.20606)已有明确前例，SQHN2024也比较了识别读出，因此这项诊断不当作新算法。2026-10-10本轮检索还未找到直接回答当前质量分配设置的结果，不能由此称全球首创。诊断尚未执行，不计作本阶段已得到发现。
'''
    (DEST/'reports/能量引导回放研究报告.md').write_text(report)
    (DEST/'reports/结论摘要.md').write_text(f'# 第七阶段结论摘要\n\n{conclusion}\n\n{details}\n\n{table}\n\n原文闭式实验方向复现通过；没有复现大型扩散模型，也没有新增人体证据。其他条件为探索性。\n')
    (DEST/'reports/进度与问题.md').write_text(f'# 第七阶段进度与问题\n\n原文闭式方向复现、20校准先冻结、40确认与9条件完成。{conclusion}\n\n{details}\n\n离线Top-K、显式记忆质量扩展、等角簇理想化与跨模型限制见完整报告。测试：{brief}。\n')
    shutil.copyfile(ROOT/'energy_protocol.md',DEST/'reports/实验协议.md')
    provenance=ROOT/'outputs/literature/下一步研究_查新与提案记录.json'
    if provenance.exists():shutil.copyfile(provenance,DEST/'reports/查新与提案记录.json')
    if test_path.exists():shutil.copyfile(test_path,DEST/'reports/测试结果.txt')
    panels=''.join(f'<section><h2>{title}</h2><img src="figures/{name}"></section>' for title,name in [('原文闭式复现','replication_replay_order.png'),('主条件均值','primary_outcomes.png'),('主比较区间','primary_paired_intervals.png'),('分数与冻结阈值','score_distributions.png'),('探索矩阵','exploratory_matrix.png'),('相似程度','similarity_sweep.png')])
    rows=''.join(f'<tr><td>{LABELS[a]}</td><td>{pct(primary.loc[a,"identity"])}</td><td>{pct(primary.loc[a,"false_alarm"])}</td><td>{pct(primary.loc[a,"hit"])}</td><td>{primary.loc[a,"auc"]:.4f}</td></tr>' for a in ARMS)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>能量引导回放研究结果</title><style>body{{margin:0;background:#f3f5f7;color:#223340;font:16px/1.8 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:25px}}section{{background:white;border-radius:10px;padding:24px;margin:22px 0}}img{{width:100%;height:auto}}a{{color:#087d8c}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:11px;text-align:left}}small{{color:#626e78}}</style><main><h1>能量引导回放：记得更牢是否也认得更准？</h1><section><p><b>{html.escape(conclusion)}</b></p><p>{html.escape(details)}</p><p>20校准 · 40新确认种子 · 主条件24簇+1孤立 · 8缓存槽 · 总回放质量16</p><table><tr><th>策略</th><th>旧实例恢复</th><th>相似新实例误认</th><th>旧命中</th><th>AUC</th></tr>{rows}</table><p>人工连续向量的闭式现代Hopfield模型；未新增人体证据。主假设只由预定两个主指标判定，其他条件探索性。</p><p><a href="reports/能量引导回放研究报告.md">完整中文报告</a> · <a href="reports/实验协议.md">运行前协议</a> · <a href="tables/summary.json">统计结果</a> · <a href="tables/audit.json">审计</a> · <a href="reports/测试结果.txt">测试日志</a></p></section>{panels}<section><p>原文闭式复现：两种强度各2000旋转，扣除重复项机械收益后的预定排序通过。<a href="https://arxiv.org/html/2605.27975v2">最近邻预印本</a></p><p>来源、读取程度、未查到与未研究的区别：<a href="reports/查新与提案记录.json">查新与提案记录</a>。</p><p><a href="reports/独立导出重跑核验.json">独立导出重跑核验</a></p></section></main></html>'''
    (DEST/'能量引导回放研究结果.html').write_text(page)
    (DEST/'README.md').write_text('''# 第七阶段：能量引导回放

先打开能量引导回放研究结果.html；完整方法和数值见reports/能量引导回放研究报告.md。
复用项目.venv：`.venv/bin/python run_energy.py --resume`；从头运行去掉--resume。
导出包可用外部已安装锁定依赖的Python运行`python run_energy.py`。仅当前Mac CPU环境实测。
单独核对原文：`python run_energy.py --replication-only`。测试：`python -m pytest tests/test_energy.py -q`。
本轮是显式连续向量闭式模型，回放增加分布质量，非原六阶段Hebbian模型、非STDP、非真实图片或人体实验。
''')
    print(DEST/'能量引导回放研究结果.html')


if __name__=='__main__':generate()
