import os, sys
from isaacgym import gymapi
gd = int(os.environ.get("GD", "0"))
g = gymapi.acquire_gym()
p = gymapi.SimParams()
p.use_gpu_pipeline = True
p.physx.use_gpu = True
print(f"creating sim compute=0 graphics={gd} ...", flush=True)
s = g.create_sim(0, gd, gymapi.SIM_PHYSX, p)
print("CREATE_SIM_OK", s is not None, flush=True)
g.prepare_sim(s)
print("PREPARE_OK", flush=True)
