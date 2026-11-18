# -*- coding: utf-8 -*-
"""硅藻显微图像分割计数界面。

启动  python app/app.py
访问  http://127.0.0.1:5000
"""

import base64
import glob
import os
import random
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from infer import load_yolo, load_self, predict_yolo, predict_self, overlay

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
MODELS = {}
MODEL_KEYS = {}


def weight_path(name):
    if name == "yolo":
        trained = os.path.join(
            ROOT, "outputs/yolo/diatom/weights/best.pt")
        return trained if os.path.exists(trained) else os.path.join(
            ROOT, "models/yolo26n-seg.pt")
    return os.path.join(ROOT, "outputs/self/best.pth")


def get_model(name):
    p = weight_path(name)
    key = (p, os.path.getmtime(p) if os.path.exists(p) else 0)
    if name not in MODELS or MODEL_KEYS.get(name) != key:
        MODELS[name] = (load_yolo(p) if name == "yolo"
                        else load_self(p))[0]
        MODEL_KEYS[name] = key
    return MODELS[name]


def encode(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>硅藻显微图像分割计数</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6eb;--text:#1c2430;--muted:#6b7583;--accent:#1f5fbf}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{padding:20px 28px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0;font-size:19px;font-weight:600}header p{margin:4px 0 0;color:var(--muted);font-size:13px}
main{display:grid;grid-template-columns:330px 1fr;gap:18px;padding:20px 28px;max-width:1500px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:18px}
label{display:block;margin:14px 0 6px;font-weight:600;font-size:13px}
select,input[type=number]{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--text)}
input[type=range]{width:100%}
button{margin-top:14px;width:100%;padding:10px;border:0;border-radius:5px;background:var(--accent);color:#fff;font-size:14px;cursor:pointer}
button.sec{background:#eef2f8;color:var(--text);border:1px solid var(--line)}
button:hover{opacity:.9}
.hint{color:var(--muted);font-size:12px;margin-top:8px}
.count{font-size:40px;font-weight:650;color:var(--accent);line-height:1.1}
.meta{color:var(--muted);font-size:13px}
#view{max-width:100%;border:1px solid var(--line);border-radius:6px;margin-top:12px;display:none}
table{width:100%;border-collapse:collapse;margin-top:14px;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}
.err{color:#b3261e;font-size:13px;margin-top:10px}
.foot{grid-column:1/-1;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
</style></head><body>
<header><h1>硅藻显微图像分割计数</h1>
<p>单类实例分割，输出逐实例掩膜与个数。数据来自 Kaggle diatom dataset，68 个硅藻物种合并为 diatom 一类。</p></header>
<main>
<div class="card">
<label>模型</label><select id="model" onchange="setDefaultConf()">
<option value="yolo">YOLO26n-seg（对照基线）</option>
<option value="self">自建单主干三分支网络</option></select>
<label>置信度阈值 <span id="cv">0.30</span></label>
<input type="range" id="conf" min="0.05" max="0.90" step="0.05" value="0.30">
<label>最小掩膜面积（像素）</label><input type="number" id="area" value="120" min="0" step="20">
<div class="hint">阈值与面积共同排除杂散颗粒、气泡与背景纹理。置信度越高，计数越保守。</div>
<label>选择显微图像</label><input type="file" id="file" accept="image/*">
<button onclick="run()">分割并计数</button>
<button class="sec" onclick="sample()">载入测试集示例</button>
<div class="err" id="err"></div>
<div class="hint">训练与评测结果见 outputs 目录。</div>
</div>
<div class="card">
<div class="count" id="count">—</div>
<div class="meta" id="meta">尚未推理。模型权重在 outputs 目录下，首次加载需要几秒。</div>
<img id="view" alt="分割结果">
<table id="tbl" style="display:none"><thead><tr><th>编号</th><th>置信度</th><th>掩膜面积</th><th>框坐标</th></tr></thead><tbody></tbody></table>
</div>
<div class="foot">口径说明。数据集标注对象为硅藻，68 个物种合并为 diatom 一类。只有被分割掩膜覆盖且通过置信度与面积过滤的目标计入个数；背景、杂质与未被标注的颗粒不计入。</div>
</main>
<script>
let sampleB64=null;
function setDefaultConf(){
  const m=document.getElementById('model').value;
  const v=(m==='yolo')?'0.30':'0.35';
  document.getElementById('conf').value=v;
  document.getElementById('cv').textContent=v;
}
document.getElementById('conf').oninput=e=>document.getElementById('cv').textContent=e.target.value;
document.getElementById('file').onchange=()=>sampleB64=null;
setDefaultConf();
async function sample(){
  const r=await fetch('/sample'); const j=await r.json();
  sampleB64=j.image; document.getElementById('err').textContent='已载入示例 '+j.filename;
  document.getElementById('view').src='data:image/jpeg;base64,'+j.image;
  document.getElementById('view').style.display='block';
}
async function run(){
  const err=document.getElementById('err'); err.textContent='';
  const fd=new FormData();
  const f=document.getElementById('file').files[0];
  if(f) fd.append('file',f); else if(sampleB64) fd.append('image',sampleB64);
  else {err.textContent='请先选择图像或载入示例';return;}
  fd.append('model',document.getElementById('model').value);
  fd.append('conf',document.getElementById('conf').value);
  fd.append('area',document.getElementById('area').value);
  document.getElementById('meta').textContent='推理中…';
  try{
    const r=await fetch('/predict',{method:'POST',body:fd}); const j=await r.json();
    if(j.error){err.textContent=j.error;document.getElementById('meta').textContent='';return;}
    document.getElementById('count').textContent=j.count;
    document.getElementById('meta').textContent=j.model+' · 置信度 '+j.conf.toFixed(2)+' · 最小面积 '+j.min_area+' 像素 · 图像 '+j.width+'×'+j.height;
    document.getElementById('view').src='data:image/jpeg;base64,'+j.image;
    document.getElementById('view').style.display='block';
    const tb=document.querySelector('#tbl tbody'); tb.innerHTML='';
    j.instances.forEach(x=>{const tr=document.createElement('tr');
      tr.innerHTML='<td>'+x.id+'</td><td>'+x.score.toFixed(3)+'</td><td>'+x.area+'</td><td>'+x.box.map(v=>Math.round(v)).join(', ')+'</td>';
      tb.appendChild(tr);});
    document.getElementById('tbl').style.display=j.instances.length?'table':'none';
  }catch(e){err.textContent='请求失败 '+e;document.getElementById('meta').textContent='';}
}
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/sample")
def sample():
    files = sorted(glob.glob(os.path.join(
        ROOT, "data_cache/images/test", "*.jpg")))
    if not files:
        return jsonify({"error": "测试集缓存尚未生成"})
    p = random.choice(files)
    img = cv2.imread(p)
    return jsonify({"image": encode(img), "filename": os.path.basename(p)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        name = request.form.get("model", "yolo")
        if name == "yolo" and not os.path.exists(os.path.join(
                ROOT, "outputs/yolo/diatom/weights/best.pt")):
            return jsonify({"error": "YOLO26 对照模型仍在训练，完成后自动生效；自建网络可立即使用"})
        conf = float(request.form.get("conf", 0.30))
        min_area = int(request.form.get("area", 120))
        raw = None
        if "file" in request.files:
            raw = request.files["file"].read()
        elif request.form.get("image"):
            raw = base64.b64decode(request.form["image"])
        if not raw:
            return jsonify({"error": "没有收到图像"})
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({"error": "图像无法解码"})
        h, w = bgr.shape[:2]
        model = get_model(name)
        if name == "yolo":
            res = predict_yolo(model, bgr, conf=conf, min_area=min_area)
        else:
            res = predict_self(model, bgr, conf=conf, min_area=min_area)
        instances = [
            {"id": i + 1, "score": round(res["scores"][i], 3),
             "area": int(res["masks"][i].sum()),
             "box": [round(v, 1) for v in res["boxes"][i]]}
            for i in range(res["count"])]
        return jsonify({
            "count": res["count"], "instances": instances,
            "image": encode(overlay(bgr, res)),
            "model": "YOLO26n-seg" if name == "yolo" else "自建三分支网络",
            "conf": conf, "min_area": min_area, "width": w, "height": h})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
