import av, numpy as np, json
from pathlib import Path
paths=['outputs/validation_oneclick_20260928/h3_base.mp4','outputs/validation_h3_diagnosis/independent_audio_seed42.mp4','outputs/validation_official_h3/official_sglang_seed42.mp4']
def frames(p):
    with av.open(p) as c:
        for f in c.decode(video=0): yield f.to_ndarray(format='rgb24').astype(np.float32)
def compare(a,b):
    total=0; absolute=0.;square=0.;count=0
    for x,y in zip(frames(a),frames(b),strict=True):
        d=x-y; absolute+=np.abs(d).sum(dtype=np.float64);square+=(d*d).sum(dtype=np.float64);total+=d.size;count+=1
    return dict(frames=count,pixel_mae_255=absolute/total,pixel_rmse_255=(square/total)**.5)
r={'old_vs_official':compare(paths[0],paths[2]),'independent_audio_vs_official':compare(paths[1],paths[2]),'old_vs_independent_audio':compare(paths[0],paths[1])}
Path('outputs/validation_h3_diagnosis/comparison.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
