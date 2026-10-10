"""Offline FR validation-only scaling comparison; no training or production edits."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLCONFIGDIR='/tmp/castmind-mpl')
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from castmind.models.base import DLinearModel, configure_deep_learning_runtime
from castmind.eval import mse, mae, smape


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    out = ROOT/'outputs/diagnostics'/('EPF_FR_dlinear_validation_'+now.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    record = ROOT/'毕业论文/04_研究记录/实验记录'/(out.name+'.md')
    record.write_text(f'''# FR DLinear 验证集标准化对照

- 时间：{now.isoformat()}；状态：运行中。用户授权独立诊断；不追查旧训练。
- 归属：探索性推理预处理对照，不作为独立泛化证据；旧权重是否使用过该验证段未知。
- 命令：.venv/bin/python scripts/diagnose_fr_dlinear_validation.py
- 数据：当前train.csv的Price仅计算固定统计量；val.csv滚动评测；不读取测试集或历史预测。
- 协议：L=168/H=24/stride=24，验证集最初168点仅作历史，每窗仅使用预测起点以前已观测值；不足24点尾窗不计。
- 条件：当前FR DLinear权重，CPU4线程、随机种子0；无训练、案例库调用、LLM、提示词或人类反馈。
- 处理：原始价格输入，对照训练集float32均值及std+1e-6标准化再还原；沿用上轮预先固定规则，不拟合验证集。
- 来源、代码版本、已有工作区改动、输入权重哈希与源码快照：{out}/provenance.json及source/。
- 产物：{out}；旧文件保持不变；本次不修改生产配置。
''')
    print('OUTPUT', out, flush=True)
    try:
        started=time.monotonic()
        torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
        train=ROOT/'data/EPF_FR/train.csv';val=ROOT/'data/EPF_FR/val.csv'
        ckpt=ROOT/'castmind/DeepLearningCheckpoints/EPF_FR/DLinear.pth'
        sources=['scripts/diagnose_fr_dlinear_validation.py','castmind/models/base.py','castmind/DeepLearningModels/DLinear.py','castmind/layers/Autoformer_EncDec.py','castmind/eval.py']
        inputs=[train,val,ckpt]+[ROOT/p for p in sources]
        hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs}
        for rel in sources:
            dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        prov=dict(hashes=hashes,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,seed=0,threads=4,device='cpu',protocol=dict(lookback=168,horizon=24,stride=24,warmup=168))
        (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2))
        tr=pd.read_csv(train,parse_dates=['date']);va=pd.read_csv(val,parse_dates=['date'])
        for d in [tr,va]:
            assert d.date.is_monotonic_increasing and d.date.is_unique
            assert (d.date.diff().dropna()==pd.Timedelta(hours=1)).all()
            assert np.isfinite(d.Price.to_numpy()).all()
        assert tr.date.max()<va.date.min()
        ytr=tr.Price.to_numpy(dtype=np.float32)
        mean,std=float(ytr.mean()),float(ytr.std()+1e-6)
        configure_deep_learning_runtime({'DLinear':str(ckpt)},24)
        state=torch.load(ckpt,map_location='cpu',weights_only=True);state=state.get('state_dict',state)
        assert list(state['Linear_Seasonal.weight'].shape)==[24,168]
        assert list(state['Linear_Trend.weight'].shape)==[24,168]
        rows=[];windows=[]
        for off in range(0,len(va)-168-24+1,24):
            past=va.iloc[off:off+168];future=va.iloc[off+168:off+192];predictions={}
            for key,scale in [('raw',False),('scaled_restored',True)]:
                x=past.Price.to_numpy(dtype=np.float32)
                if scale:x=(x-mean)/std
                model=DLinearModel();model.fit(x,season_length=24,timestamps=past.date)
                model._model.load_state_dict(state,strict=True)
                p=model.predict(24,future_timestamps=future.date)
                if scale:p=p*std+mean
                assert p.shape==(24,) and np.isfinite(p).all()
                predictions[key]=p.astype(float)
            win=dict(window_offset=off,start_timestamp=str(future.date.iloc[0]))
            for k,p in predictions.items():
                win[k+'_MSE']=mse(future.Price.to_numpy(),p);win[k+'_MAE']=mae(future.Price.to_numpy(),p)
            windows.append(win)
            for i in range(24):
                rows.append(dict(time_stamp=str(future.date.iloc[i]),window_offset=off,horizon_index=i,truth=float(future.Price.iloc[i]),**{k:float(p[i]) for k,p in predictions.items()}))
        d=pd.DataFrame(rows);w=pd.DataFrame(windows)
        assert len(d)>0 and not d.time_stamp.duplicated().any()
        assert pd.to_datetime(d.time_stamp).tolist()==va.date.iloc[168:168+len(d)].tolist()
        metrics={k:dict(MSE=mse(d.truth.to_numpy(),d[k].to_numpy()),MAE=mae(d.truth.to_numpy(),d[k].to_numpy()),sMAPE=smape(d.truth.to_numpy(),d[k].to_numpy())) for k in ['raw','scaled_restored']}
        delta=w.scaled_restored_MSE-w.raw_MSE
        summary=dict(metrics=metrics,mean=mean,std=std,train_rows=len(tr),val_rows=len(va),windows=len(w),points=len(d),warmup_points=168,tail_points=len(va)-168-len(d),start=d.time_stamp.iloc[0],end=d.time_stamp.iloc[-1],improved_windows=int((delta<0).sum()),worse_windows=int((delta>0).sum()),tied_windows=int((delta==0).sum()),median_window_MSE_delta=float(delta.median()),strict_load=True,inputs_unchanged=all(digest(p)==hashes[str(p.relative_to(ROOT))] for p in inputs),elapsed_seconds=time.monotonic()-started,llm_calls=0,scope='Exploratory validation segment; independence from historical model training unknown; no claim of Full improvement')
        d.to_csv(out/'predictions.csv',index=False);w.to_csv(out/'window_metrics.csv',index=False)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        # Independent check against persisted predictions, not just in-memory vectors.
        saved=pd.read_csv(out/'predictions.csv')
        for k in metrics:
            assert np.isclose(np.mean((saved[k]-saved.truth)**2),metrics[k]['MSE'],rtol=1e-12)
            assert np.isclose(np.mean(abs(saved[k]-saved.truth)),metrics[k]['MAE'],rtol=1e-12)
        record.write_text(record.read_text()+f'\n## 完成\n- 时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}。\n- {len(w)}窗、{len(d)}唯一时点；持久化预测独立重算MSE/MAE通过；输入哈希未变：{summary["inputs_unchanged"]}。\n- 结果：{json.dumps(metrics,ensure_ascii=False)}。\n- 改善/恶化/持平窗口：{summary["improved_windows"]}/{summary["worse_windows"]}/{summary["tied_windows"]}。\n- 核心流程耗时{summary["elapsed_seconds"]:.3f}秒，不含依赖导入；无LLM/API费用。不改变旧实验判断，不宣称泛化或完整系统收益。\n')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    except Exception as exc:
        record.write_text(record.read_text()+f'\n状态：失败；{type(exc).__name__}: {exc}\n');raise


if __name__=='__main__':
    main()
