"""Paired post-hoc FR fusion diagnostic using saved forecasts only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metric(y,p):
    e=p-y;den=np.abs(y)+np.abs(p)
    return dict(MSE=float(np.mean(e**2)),MAE=float(np.mean(abs(e))),sMAPE=float(np.mean(2*abs(e)/np.where(den==0,1,den))))

def main():
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    out=ROOT/'outputs/diagnostics'/('EPF_FR_fusion_dlinear_swap_'+now.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    record=ROOT/'毕业论文/04_研究记录/实验记录'/(out.name+'.md')
    record.write_text(f'''# FR 融合单成员替换诊断

- 时间：{now.isoformat()}；状态：运行中。
- 用户授权：独立实验替换DLinear成员预测，其他成员、权重不变。不改生产、不追查旧训练。
- 命令：.venv/bin/python scripts/diagnose_fr_fusion_dlinear_swap.py
- 数据：已有FR测试段119窗2856点，非验证段。本轮事后诊断，不声称独立泛化验证。
- 条件：L=168/H=24/stride=24；复用已有两种Sundial种子。DLinear使用前轮固定训练集统计量标准化预测。
- 不调用模型、训练或LLM；无API费用、提示词或人类反馈；案例选择及成员权重完全固定。
- 原始实验文件只读；来源及哈希、代码版本和已有工作区改动见{out}/provenance.json。
- 产物：{out}。不能把重建辅助收益当成历史Full收益。
''')
    try:
        fusion=ROOT/'outputs/diagnostics/EPF_FR_fusion_check_20261009_175253'
        scaling=ROOT/'outputs/diagnostics/EPF_FR_dlinear_scaling_20261010_100206'
        paths=[fusion/'aligned_predictions.csv',fusion/'member_predictions.jsonl',fusion/'provenance.json',scaling/'predictions.csv',scaling/'provenance.json',scaling/'summary.json',Path(__file__)]
        hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
        (out/'provenance.json').write_text(json.dumps(dict(hashes=hashes,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),numpy=np.__version__,pandas=pd.__version__),ensure_ascii=False,indent=2))
        shutil.copy2(__file__,out/'recompute.py')
        fp=json.loads((fusion/'provenance.json').read_text())['hashes'];sp=json.loads((scaling/'provenance.json').read_text())['hashes']
        shared=set(fp)&set(sp);assert shared and all(fp[k]==sp[k] for k in shared)
        f=pd.read_csv(paths[0]);s=pd.read_csv(scaling/'predictions.csv')
        keys=['time_stamp','window_offset','horizon_index']
        d=f.merge(s[keys+['truth','raw','scaled_restored']],on=keys,validate='one_to_one',suffixes=('','_scaling'))
        assert len(d)==len(f)==len(s)==2856 and d.time_stamp.nunique()==2856
        np.testing.assert_allclose(d.truth,d.truth_scaling,rtol=0,atol=0)
        weight=d.weight_DLinear.fillna(0);mask=weight>0
        np.testing.assert_allclose(d.loc[mask,'raw'],d.loc[mask,'member_DLinear'],rtol=0,atol=1e-12)
        records=[json.loads(x) for x in paths[1].read_text().splitlines()]
        grouped={}
        for r in records:grouped.setdefault(r['window_offset'],[]).append(r)
        windows=[]
        for seed,col in [(0,'reconstructed'),(1,'reconstructed_seed1')]:
            newcol=f'swapped_seed{seed}'
            d[newcol]=d[col]+weight*(d.scaled_restored-d.raw)
            for off,g in d.groupby('window_offset',sort=True):
                rs=grouped[off];assert abs(sum(r['weight'] for r in rs)-1)<1e-12
                old=np.zeros(24);new=np.zeros(24)
                for r in rs:
                    p=np.array(r.get('seed1_prediction',r['prediction']) if seed else r['prediction'])
                    old+=r['weight']*p
                    new+=r['weight']*(g.scaled_restored.to_numpy() if r['model']=='DLinear' else p)
                np.testing.assert_allclose(old,g[col],rtol=0,atol=1e-10)
                np.testing.assert_allclose(new,g[newcol],rtol=0,atol=1e-10)
                oldm=metric(g.truth.to_numpy(),old);newm=metric(g.truth.to_numpy(),new)
                windows.append(dict(seed=seed,window_offset=int(off),dlinear_present=bool((g.weight_DLinear.fillna(0)>0).any()),before_MSE=oldm['MSE'],after_MSE=newm['MSE'],delta_MSE=newm['MSE']-oldm['MSE']))
            np.testing.assert_array_equal(d.loc[~mask,newcol],d.loc[~mask,col])
        metrics={k:metric(d.truth.to_numpy(),d[k].to_numpy()) for k in ['chronos','auxiliary','full','reconstructed','reconstructed_seed1','swapped_seed0','swapped_seed1']}
        w=pd.DataFrame(windows);paired={}
        for seed in [0,1]:
            z=w[w.seed==seed];a=metrics['reconstructed' if seed==0 else 'reconstructed_seed1'];b=metrics[f'swapped_seed{seed}']
            paired[str(seed)]=dict(MSE_change_percent=100*(b['MSE']/a['MSE']-1),MAE_change_percent=100*(b['MAE']/a['MAE']-1),improved=int((z.delta_MSE < -1e-10).sum()),worse=int((z.delta_MSE > 1e-10).sum()),tied=int((abs(z.delta_MSE)<=1e-10).sum()),MSE_vs_chronos_percent=100*(b['MSE']/metrics['chronos']['MSE']-1))
        summary=dict(metrics=metrics,paired=paired,points=len(d),windows=119,dlinear_windows=int(d.loc[mask,'window_offset'].nunique()),unchanged_windows=int(d.loc[~mask,'window_offset'].nunique()),shared_provenance_hashes_matched=len(shared),inputs_unchanged=all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in paths),model_calls=0,llm_calls=0,scope='Post-hoc test-segment reconstruction; not Full prediction, validation experiment, or independent generalization evidence')
        d.to_csv(out/'predictions.csv',index=False);w.to_csv(out/'window_metrics.csv',index=False)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        lines=['# FR 辅助融合：仅替换DLinear','', '仅复用既有测试段成员预测，固定案例选择、所有权重和其他成员，包括两种Sundial随机种子。不是新验证集实验，也没有改动LLM最终预测。','']
        for seed in [0,1]:
            a=metrics['reconstructed' if seed==0 else 'reconstructed_seed1'];b=metrics[f'swapped_seed{seed}'];p=paired[str(seed)]
            lines.append(f"- 种子{seed}：MSE {a['MSE']:.6f} → {b['MSE']:.6f}（{p['MSE_change_percent']:+.2f}%）；MAE {a['MAE']:.6f} → {b['MAE']:.6f}。窗口改善/恶化/持平：{p['improved']}/{p['worse']}/{p['tied']}。替换后MSE相对同段Chronos {p['MSE_vs_chronos_percent']:+.2f}%。")
        lines+=['',f"同一时点Chronos：MSE {metrics['chronos']['MSE']:.6f}，MAE {metrics['chronos']['MAE']:.6f}。",'', '## 核验与解释','', '覆盖119窗2856唯一时点，真实值与两组文件一一对齐；102窗含DLinear，另外17窗保持不变。直接逐模型重组与差量替换两种算法逐点吻合。DLinear原始预测与既有成员一致；共享输入/源码哈希吻合，原始文件未变化。','', '替换公式：新融合 = 旧重建融合 + DLinear原权重 ×（标准化后DLinear − 原DLinear）。不增加或删除成员，不重新归一化权重。','', '完整历史辅助预测不能精确回放，所以只以当前离线重建的配对差值判断影响，不能与历史辅助33.27直接相减宣称修复收益。标准化规则此前已查看测试结果，本轮只作机制诊断。旧权重是否用过验证段仍未知，不追查。','', '未修改生产预测代码、旧归档、权重或案例库；无推理、训练、LLM调用。进一步实验或接入需用户另行批准。','', '来源、数据和代码条件见provenance.json；精确指标见summary.json；逐点与逐窗见predictions.csv、window_metrics.csv。复跑入口为仓库scripts/diagnose_fr_fusion_dlinear_swap.py，自动创建新的输出目录。']
        (out/'README.md').write_text('\n'.join(lines)+'\n')
        record.write_text(record.read_text()+f'\n## 完成\n- {datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}：119窗2856点；102窗替换、17窗不变；直接重组交叉核验通过。\n- 配对结果：{json.dumps(paired,ensure_ascii=False)}。\n- 原文件哈希不变：{summary["inputs_unchanged"]}。结论范围限于离线重建辅助，不代表Full或独立泛化。\n')
        print('OUTPUT',out);print(json.dumps(summary,ensure_ascii=False,indent=2))
    except Exception as exc:
        record.write_text(record.read_text()+f'\n失败：{type(exc).__name__}: {exc}\n');raise

if __name__=='__main__':main()
