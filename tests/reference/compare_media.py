"""Compare decoded pixels/samples, independent of container metadata."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('official');p.add_argument('local');p.add_argument('--output',required=True);args=p.parse_args()
report={}
for stream,dtype,fmt in [('video',np.uint8,'rawvideo'),('audio',np.float32,'f32le')]:
 values=[]
 for path in [args.official,args.local]:
  cmd=['ffmpeg','-v','error','-i',path,'-map','0:'+stream[0]]
  if stream=='video':cmd+=['-pix_fmt','rgb24']
  raw=subprocess.check_output(cmd+['-f',fmt,'-']);values.append(np.frombuffer(raw,dtype=dtype))
 a,b=values;r={'official_elements':a.size,'local_elements':b.size,'official_sha256':hashlib.sha256(a).hexdigest(),'local_sha256':hashlib.sha256(b).hexdigest()}
 if a.shape==b.shape:
  d=a.astype(np.float64)-b.astype(np.float64);r.update(exact=bool(np.array_equal(a,b)),mae=float(np.abs(d).mean()),rmse=float(np.sqrt(np.mean(d*d))),max_abs=float(np.abs(d).max()))
 report[stream]=r
Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
