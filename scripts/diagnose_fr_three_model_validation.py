"""Offline FR validation comparison. Fixed equal weights; no tuning or training."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR='/tmp/castmind-mpl')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil,subprocess,sys,time,random
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from castmind.models.base import ChronosModel,SundialModel,TimesNetModel,configure_deep_learning_runtime
from castmind.eval import mse,mae,smape

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def metrics(y,p):
    return dict(MSE=mse(y,p),MAE=mae(y,p),sMAPE=smape(y,p))

def main():
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    out=ROOT/'outputs/diagnostics'/('EPF_FR_three_model_validation_'+now.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',out,flush=True)
    record=ROOT/'毕业论文/04_研究记录/实验记录'/(out.name+'.md')
    record.write_text(f'''# FR 验证段：三模型及等权平均

- 时间：{now.isoformat()}；状态：运行中。用户授权本地独立诊断。
- 命令：.venv/bin/python scripts/diagnose_fr_three_model_validation.py
- 预先固定：Chronos、Sundial、TimesNet三个单模型，以及1/3等权平均；不选择权重、不优化超参。
- 数据：仅当前data/EPF_FR/val.csv，目标Price。L=168/H=24/stride=24，起始168点作历史，不足24点尾窗丢弃。预计59窗1416点。
- 模型：本地Chronos bolt-base、Sundial base-128m及FR TimesNet权重，沿用当前推理接口；TimesNet不增加外部标准化。基础模型缓存只读权重，逐窗更新输入；TimesNet每窗新建并严格加载。
- Sundial具有随机采样：主结果种子0，预先固定种子1敏感性对照；每窗预测前重设种子，不择优报告。
- CPU4线程；无重训、案例库、LLM或提示词，不追查旧训练；旧权重与验证段关系未知，非独立泛化证明。
- 来源与哈希、代码版本和已有工作区改动见{out}/provenance.json；源码快照source/。
- 产物：{out}；旧归档和生产逻辑不变，无反馈参与预测。
''')
    try:
        started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0)
        assert not torch.cuda.is_available(), 'This diagnostic protocol is CPU-only.'
        val=ROOT/'data/EPF_FR/val.csv';ckpt=ROOT/'castmind/DeepLearningCheckpoints/EPF_FR/TimesNet.pth'
        source=[Path(__file__),ROOT/'castmind/models/base.py',ROOT/'castmind/eval.py']
        source+=list((ROOT/'castmind/DeepLearningModels').glob('*.py'))+list((ROOT/'castmind/layers').glob('*.py'))+list((ROOT/'castmind/utils').glob('*.py'))
        files=[val,ckpt]+source
        for name in ['chronos-bolt-base','sundial-base-128m']:
            folder=ROOT/'castmind/foundation_models'/name;assert folder.is_dir()
            files += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.json','.safetensors','.bin','.py']]
        hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
        for p in source:
            dst=out/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        prov=dict(hashes=hashes,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,preset_override=os.getenv('CASTMIND_DL_PRESET'),seeds=[0,1],threads=4,device='cpu',weights=[1/3]*3)
        (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2))
        va=pd.read_csv(val,parse_dates=['date']);assert va.date.is_unique and va.date.is_monotonic_increasing
        assert (va.date.diff().dropna()==pd.Timedelta(hours=1)).all() and np.isfinite(va.Price).all()
        configure_deep_learning_runtime({'TimesNet':str(ckpt)},24)
        state=torch.load(ckpt,map_location='cpu',weights_only=True);state=state.get('state_dict',state)
        chronos=ChronosModel(local_dir=str(ROOT/'castmind/foundation_models/chronos-bolt-base'))
        sundial=SundialModel();sundial.local_dir=str(ROOT/'castmind/foundation_models/sundial-base-128m')
        # Load before resetting forecast seeds, so loading RNG use cannot affect forecasts.
        chronos._ensure_pipeline();sundial._ensure_model()
        rows=[];wins=[]
        with (out/'window_predictions.jsonl').open('w') as stream:
            for wi,off in enumerate(range(0,len(va)-168-24+1,24)):
                past=va.iloc[off:off+168];future=va.iloc[off+168:off+192];pred={}
                timesnet=TimesNetModel()
                for name,model in [('chronos',chronos),('timesnet',timesnet)]:
                    torch.manual_seed(0);np.random.seed(0);random.seed(0)
                    model.fit(past.Price.to_numpy(),season_length=24,timestamps=past.date)
                    if name=='timesnet':model._model.load_state_dict(state,strict=True)
                    pred[name]=np.asarray(model.predict(24,future_timestamps=future.date),dtype=float).reshape(-1)
                sundial.fit(past.Price.to_numpy(),season_length=24,timestamps=past.date)
                for seed in [0,1]:
                    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                    pred[f'sundial_seed{seed}']=np.asarray(sundial.predict(24,future_timestamps=future.date),dtype=float).reshape(-1)
                    pred[f'equal_seed{seed}']=(pred['chronos']+pred['timesnet']+pred[f'sundial_seed{seed}'])/3
                assert all(p.shape==(24,) and np.isfinite(p).all() for p in pred.values())
                stream.write(json.dumps(dict(window_offset=off,predictions={k:p.tolist() for k,p in pred.items()}))+'\n');stream.flush()
                w=dict(window_offset=off,start_timestamp=str(future.date.iloc[0]))
                for k,p in pred.items():
                    for m,v in metrics(future.Price.to_numpy(),p).items():w[k+'_'+m]=v
                wins.append(w)
                for i in range(24):rows.append(dict(time_stamp=str(future.date.iloc[i]),window_offset=off,horizon_index=i,truth=float(future.Price.iloc[i]),**{k:float(p[i]) for k,p in pred.items()}))
                if (wi+1)%10==0:print('completed',wi+1,flush=True)
        d=pd.DataFrame(rows);w=pd.DataFrame(wins);assert len(d)==1416 and len(w)==59 and d.time_stamp.is_unique
        assert pd.to_datetime(d.time_stamp).tolist()==va.date.iloc[168:168+len(d)].tolist()
        scores={k:metrics(d.truth.to_numpy(),d[k].to_numpy()) for k in pred}
        comparison={}
        for k in pred:
            delta=w[k+'_MSE']-w.chronos_MSE
            comparison[k]=dict(MSE_vs_chronos_percent=100*(scores[k]['MSE']/scores['chronos']['MSE']-1),MAE_vs_chronos_percent=100*(scores[k]['MAE']/scores['chronos']['MAE']-1),better_windows=int((delta < -1e-10).sum()),worse_windows=int((delta > 1e-10).sum()),tie_windows=int((abs(delta)<=1e-10).sum()),median_window_MSE_delta=float(delta.median()))
        d.to_csv(out/'predictions.csv',index=False);w.to_csv(out/'window_metrics.csv',index=False)
        saved=pd.read_csv(out/'predictions.csv')
        for k in pred:
            np.testing.assert_allclose(scores[k]['MSE'],np.mean((saved[k]-saved.truth)**2),rtol=1e-12)
            np.testing.assert_allclose(scores[k]['MAE'],np.mean(abs(saved[k]-saved.truth)),rtol=1e-12)
        for seed in [0,1]:np.testing.assert_allclose(saved[f'equal_seed{seed}'],(saved.chronos+saved.timesnet+saved[f'sundial_seed{seed}'])/3,atol=1e-12,rtol=0)
        summary=dict(metrics=scores,comparison=comparison,points=len(d),windows=len(w),start=d.time_stamp.iloc[0],end=d.time_stamp.iloc[-1],warmup=168,tail=len(va)-168-len(d),inputs_unchanged=all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in files),strict_timesnet_load=True,elapsed_seconds=time.monotonic()-started,model_forecast_calls=4*len(w),llm_calls=0,scope='Exploratory validation-segment comparison; no proof of independence from prior training; no Full experiment')
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        record.write_text(record.read_text()+f'\n## 完成\n- {datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}：59窗1416点，逐点等权公式及落盘指标复算通过。\n- 结果：{json.dumps(scores,ensure_ascii=False)}。\n- 与Chronos比较：{json.dumps(comparison,ensure_ascii=False)}。\n- 原文件哈希不变：{summary["inputs_unchanged"]}。无LLM/API费用；核心流程{summary["elapsed_seconds"]:.2f}秒，不含导入。\n- 仅当前验证段探索结果，不代表Full或跨数据集有效性；不自动运行下一轮。\n')
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        record.write_text(record.read_text()+f'\n失败：{type(exc).__name__}: {exc}\n');raise

if __name__=='__main__':main()
