"""Isolated, offline scaling sensitivity check; never edits model code or archives."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    out = args.output or ROOT / 'outputs/diagnostics' / ('EPF_FR_dlinear_scaling_' + now.strftime('%Y%m%d_%H%M%S'))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    record = ROOT / '毕业论文/04_研究记录/实验记录' / (out.name + '.md')
    record.write_text(f'''# FR DLinear 标准化敏感性诊断

- 时间：{now.isoformat()}，Asia/Shanghai；状态：运行中。
- 用户授权：独立诊断脚本及本地对照，不修改生产预测逻辑、重训或调用LLM。
- 归属：本地诊断，非算法有效性实验；变化因素：输入标准化及输出还原。
- 命令：.venv/bin/python scripts/diagnose_fr_dlinear_scaling.py --output {out}
- 数据：data/EPF_FR/train.csv 仅计算均值标准差；test.csv评测原始Price，168/24/24。
- 当前训练集统计量只是候选处理，不证明等于旧权重训练统计量。旧训练来源待核验。
- 模型：当前FR DLinear权重；CPU，4线程，种子0，不更新权重，无LLM和提示词。
- 案例库：不调用、不修改；原始归档只读。
- 代码、已有修改、依赖、权重和输入哈希：见产物provenance.json及source/。
- 产物：{out}；失败或完成后补充。无用户反馈参与模型。
''')
    try:
        started = time.monotonic()
        torch.set_num_threads(4)
        torch.manual_seed(0)
        np.random.seed(0)
        run = ROOT / 'outputs/EPF_FR/runs/Full_1'
        trainfile = ROOT / 'data/EPF_FR/train.csv'
        testfile = ROOT / 'data/EPF_FR/test.csv'
        checkpoint = ROOT / 'castmind/DeepLearningCheckpoints/EPF_FR/DLinear.pth'
        history = ROOT / 'outputs/diagnostics/EPF_FR_fusion_check_20261009_175253/member_predictions.jsonl'
        source = ['castmind/models/base.py', 'castmind/DeepLearningModels/DLinear.py', 'castmind/layers/Autoformer_EncDec.py', 'castmind/eval.py', 'scripts/train_checkpoints.py', 'scripts/diagnose_fr_dlinear_scaling.py']
        inputs = [trainfile, testfile, checkpoint, run/'predictions.csv', history] + [ROOT / x for x in source]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
        for rel in source:
            dest = out/'source'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, dest)
        prov = dict(hashes=hashes, git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, device='cpu', threads=4, seed=0, llm_calls=0)
        (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2))
        tr = pd.read_csv(trainfile)
        te = pd.read_csv(testfile,parse_dates=['date'])
        full = pd.read_csv(run/'predictions.csv',parse_dates=['time_stamp'])
        ytrain = pd.to_numeric(tr.Price).to_numpy(dtype=np.float32)
        ytrain = ytrain[np.isfinite(ytrain)]
        mean, std = float(np.mean(ytrain)), float(np.std(ytrain)+1e-6)
        assert std > 0 and pd.to_datetime(tr.date).max() < te.date.min()
        configure_deep_learning_runtime({'DLinear': str(checkpoint)},24)
        state = torch.load(checkpoint,map_location='cpu',weights_only=True)
        state = state.get('state_dict',state)
        shapes = {k:list(v.shape) for k,v in state.items()}
        assert shapes['Linear_Seasonal.weight'] == [24,168]
        assert shapes['Linear_Trend.weight'] == [24,168]
        (out/'checkpoint_structure.json').write_text(json.dumps(shapes,indent=2))
        rows=[]
        for off,g in full.groupby('window_offset',sort=True):
            off=int(off);g=g.sort_values('horizon_index');past=te.iloc[off:off+168];future=te.iloc[off+168:off+192]
            assert len(g)==24 and g.time_stamp.tolist()==future.date.tolist()
            predictions={}
            for label,scaled in [('raw',False),('scaled_restored',True)]:
                x=past.Price.to_numpy(dtype=np.float32)
                if scaled:x=(x-mean)/std
                model=DLinearModel()
                model.fit(x,season_length=24,timestamps=past.date)
                model._model.load_state_dict(state,strict=True)
                pred=model.predict(24,future_timestamps=future.date)
                if scaled:pred=pred*std+mean
                assert pred.shape==(24,) and np.isfinite(pred).all()
                predictions[label]=pred
            for i in range(24):
                rows.append(dict(time_stamp=str(future.date.iloc[i]),window_offset=off,horizon_index=i,truth=float(future.Price.iloc[i]),raw=float(predictions['raw'][i]),scaled_restored=float(predictions['scaled_restored'][i])))
        df=pd.DataFrame(rows)
        assert len(df)==2856 and df.time_stamp.nunique()==2856
        df.to_csv(out/'predictions.csv',index=False)
        metrics={k:dict(MSE=mse(df.truth.to_numpy(),df[k].to_numpy()),MAE=mae(df.truth.to_numpy(),df[k].to_numpy()),sMAPE=smape(df.truth.to_numpy(),df[k].to_numpy())) for k in ['raw','scaled_restored']}
        windows=[]
        for off,g in df.groupby('window_offset'):
            windows.append(dict(window_offset=int(off),raw_MSE=mse(g.truth.to_numpy(),g.raw.to_numpy()),scaled_MSE=mse(g.truth.to_numpy(),g.scaled_restored.to_numpy())))
        pd.DataFrame(windows).to_csv(out/'window_metrics.csv',index=False)
        prior=[json.loads(line) for line in history.read_text().splitlines()]
        differences=[float(np.max(abs(df.loc[df.window_offset==r['window_offset'],'raw'].to_numpy()-np.array(r['prediction'])))) for r in prior if r['model']=='DLinear']
        summary=dict(metrics=metrics,train_mean=mean,train_std=std,train_points=len(ytrain),windows=len(windows),points=len(df),strict_checkpoint_load=True,checkpoint_shapes=shapes,prior_matching_windows=len(differences),prior_raw_max_abs_difference=max(differences),scaled_better_windows=sum(w['scaled_MSE']<w['raw_MSE'] for w in windows),elapsed_seconds=time.monotonic()-started,inputs_unchanged=all(digest(p)==hashes[str(p.relative_to(ROOT))] for p in inputs),historical_training_provenance='unknown; current train statistics are a hypothesis, not recovered training metadata',llm_calls=0)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        record.write_text(record.read_text()+f'\n## 完成\n\n- {datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}：覆盖119窗2856唯一时点；严格加载通过，旧输入哈希不变：{summary["inputs_unchanged"]}。\n- 原始输入与上轮DLinear重合窗口最大差异：{max(differences)}（{len(differences)}窗）。\n- 两种处理指标：{json.dumps(metrics,ensure_ascii=False)}。\n- 耗时{summary["elapsed_seconds"]:.2f}秒；无LLM/API费用。仅敏感性诊断，不能据此认证旧权重训练来源或归因完整系统退化。\n')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    except Exception as exc:
        record.write_text(record.read_text()+f'\n失败：{type(exc).__name__}: {exc}\n')
        raise


if __name__ == '__main__':
    main()
