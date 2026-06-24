import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import importlib.util, numpy as np, torch, trimesh, pyrender, imageio.v2 as imageio
import smplx, sys
sys.path.insert(0, ".")

ED = "motion_annot_web/eval_dashboard/utils.py"
spec = importlib.util.spec_from_file_location("ed", ED); ed = importlib.util.module_from_spec(spec); spec.loader.exec_module(ed)

dev = torch.device("cuda")
model = smplx.create("ref_repo/MDM/body_models_nochumpy", model_type="smpl", gender="neutral", ext="pkl", batch_size=1).to(dev).eval()

z = np.load("output/evaluation/mib_ms272_ikfix/gtctrl/smplx/000824.npz", allow_pickle=True)
m135 = z["motion_135"].astype(np.float32)  # column-major
fi = 80  # a mid frame

def reorder_col_to_row(m):
    out = m.copy(); r = m[:, 3:135].reshape(-1, 22, 6)
    r = r[:, :, [0, 3, 1, 4, 2, 5]]
    out[:, 3:135] = r.reshape(m.shape[0], 132); return out

def verts_from_poses(poses66, Th):
    go = poses66[:, :3]; bp = poses66[:, 3:66]
    b = go.shape[0]; body69 = np.zeros((b, 69), np.float32); body69[:, :63] = bp
    with torch.no_grad():
        out = model(betas=torch.zeros(b,10,device=dev), body_pose=torch.from_numpy(body69).to(dev),
                    global_orient=torch.from_numpy(go.astype(np.float32)).to(dev),
                    transl=torch.from_numpy(Th.astype(np.float32)).to(dev))
    return out.vertices[0].cpu().numpy()

def viewer_poses(m):
    r = ed._smpl_from_motion135({"motion_135": m}, "local")
    fr = r["frames"]; T = len(fr)
    poses = np.array([fr[i][0]["poses"][0][:66] for i in range(T)], np.float32)
    Th = np.array([fr[i][0]["Th"][0] for i in range(T)], np.float32)
    return poses, Th

# A: viewer path on column data (as-is)
pA, thA = viewer_poses(m135); vA = verts_from_poses(pA[fi:fi+1], thA[fi:fi+1])
# B: viewer path on reordered (col->row) data
pB, thB = viewer_poses(reorder_col_to_row(m135)); vB = verts_from_poses(pB[fi:fi+1], thB[fi:fi+1])
# GT: direct from axis-angle in npz
go = z["global_orient"][fi:fi+1]; bp = z["body_pose"][fi:fi+1]; tr = z["transl"][fi:fi+1]
b=1; body69=np.zeros((1,69),np.float32); body69[:,:63]=bp
with torch.no_grad():
    o=model(betas=torch.zeros(1,10,device=dev),body_pose=torch.from_numpy(body69).to(dev),
            global_orient=torch.from_numpy(go.astype(np.float32)).to(dev),transl=torch.from_numpy(tr.astype(np.float32)).to(dev))
vGT=o.vertices[0].cpu().numpy()

faces = model.faces.astype(np.int32)
r = pyrender.OffscreenRenderer(360, 480)
def render(v):
    v = v.copy(); v[:,1]-=v[:,1].min()
    c=np.array([v[:,0].mean(),0.9,v[:,2].mean()],np.float32)
    sc=pyrender.Scene(bg_color=[1,1,1,1],ambient_light=[0.4,0.4,0.4])
    sc.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v,faces,process=False),smooth=True))
    eye=[c[0]+2.0,c[1]+1.2,c[2]+2.3]
    f=np.array([c[0],c[1]*0.9,c[2]])-np.array(eye); f/=np.linalg.norm(f)
    s=np.cross(f,[0,1,0]); s/=np.linalg.norm(s); u=np.cross(s,f)
    cp=np.eye(4); cp[:3,0]=s; cp[:3,1]=u; cp[:3,2]=-f; cp[:3,3]=eye
    sc.add(pyrender.PerspectiveCamera(yfov=0.78,aspectRatio=0.75),pose=cp)
    sc.add(pyrender.DirectionalLight(intensity=3.0),pose=cp)
    col,_=r.render(sc); return col
img=np.concatenate([render(vA),render(vB),render(vGT)],axis=1)
imageio.imwrite("output/evaluation/mib_ms272_ikfix/_conv_test.png", img)
print("col-mpjpe vsGT  A:", np.abs(vA-vGT).mean()*1000, " B:", np.abs(vB-vGT).mean()*1000, "mm")
print("wrote _conv_test.png (left=as-is column, mid=col->row reorder, right=GT axis-angle)")
