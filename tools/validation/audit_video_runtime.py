"""Audit external model imports and probe public runtimes with them blocked."""
import ast
import importlib
import importlib.abc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {'diffusers', 'transformers', 'tokenizers', 'peft', 'sglang', 'ltx_core', 'ltx_pipelines', 'ltx_trainer'}
violations = []
files = []
for category in ('models', 'pipelines', 'trainers'):
    for model in ('minimax_h3', 'ltx_video'):
        files.extend((ROOT / 'hftrainer' / category / model).rglob('*.py'))
files.append(ROOT / 'hftrainer/tokenization.py')
for path in files:
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        names = []
        if isinstance(node, ast.Import): names = [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom) and node.level == 0: names = [node.module or '']
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
            if name in {'import_module', '__import__'}: names = [node.args[0].value]
        for name in names:
            if name.split('.')[0] in FORBIDDEN:
                violations.append({'file':str(path.relative_to(ROOT)), 'line':node.lineno, 'module':name})
class BlockExternalModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError(f'External model package blocked: {fullname}')
sys.meta_path.insert(0, BlockExternalModels())
probed = []
for name in ('hftrainer.models.minimax_h3', 'hftrainer.pipelines.minimax_h3', 'hftrainer.trainers.minimax_h3', 'hftrainer.models.ltx_video', 'hftrainer.pipelines.ltx_video', 'hftrainer.trainers.ltx_video', 'hftrainer.pipelines.ltx_video.backend.ti2vid_two_stages', 'hftrainer.trainers.ltx_video.native.trainer'):
    importlib.import_module(name); probed.append(name)
loaded = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
report = {'files_scanned':len(files),'static_violations':violations,'blocked_roots':sorted(FORBIDDEN),'successful_import_probes':probed,'external_model_modules_loaded':loaded,'scope':'Production model/pipeline/trainer paths; independent official reference tests intentionally excluded.'}
print(json.dumps(report, indent=2))
if violations or loaded: raise SystemExit(1)
