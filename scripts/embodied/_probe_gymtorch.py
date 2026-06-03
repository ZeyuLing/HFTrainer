import sys
try:
    import isaacgym
    from isaacgym import gymtorch
    import torch
    print("GYMTORCH_OK", torch.__version__, torch.cuda.is_available())
except Exception as e:
    import traceback
    traceback.print_exc()
    print("GYMTORCH_FAIL", repr(e))
    sys.exit(1)
