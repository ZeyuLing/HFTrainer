#!/usr/bin/env bash
# 在 HFTrainer 根目录执行：按 ref_repo 子项目拆分 add / commit（依赖当前 .gitignore）。
# 适用场景：已将历史 reset 到纳入 ref_repo 之前，工作区里已有完整 ref_repo 目录但未跟踪。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(git branch --show-current)" != "motion" ]]; then
  echo "请先: git checkout motion"
  exit 1
fi

if [[ -n "${SKIP_TRACKED_CHECK:-}" ]]; then
  :
elif [[ -n "$(git status --porcelain=v1 --untracked-files=no 2>/dev/null | grep -v 'scripts/ref_repo_commit_by_project.sh' || true)" ]]; then
  echo "警告：仍有已跟踪文件的修改，请先 stash/commit，或设置 SKIP_TRACKED_CHECK=1 跳过检查。"
  git status -s | head -25
  exit 1
fi

echo ">>> 1/3 chore(gitignore): .tar.xz"
git add .gitignore
git commit -m "chore(gitignore): ref_repo 补充 .tar.xz / .xz（避免压缩包漏网）"

echo ">>> 2/3 根目录文档"
roots=()
[[ -f ref_repo/CLAUDE.md ]] && roots+=(ref_repo/CLAUDE.md)
[[ -f ref_repo/m2m_ablation_experiments.md ]] && roots+=(ref_repo/m2m_ablation_experiments.md)
if ((${#roots[@]})); then
  git add "${roots[@]}"
  if ! git diff --cached --quiet; then
    git commit -m "feat(ref_repo): 根目录 CLAUDE 与 m2m_ablation_experiments"
  fi
fi

echo ">>> 3/3 各子项目（按目录名排序）"
mapfile -t PROJS < <(find ref_repo -mindepth 1 -maxdepth 1 \( -type d -o -type l \) -printf '%f\n' | LC_ALL=C sort)

for d in "${PROJS[@]}"; do
  case "$d" in
    *.zip|*.rar) echo "跳过归档: $d"; continue ;;
  esac
  [[ -e "ref_repo/$d" ]] || continue
  echo "---- ref_repo/$d ----"
  git add "ref_repo/$d/"
  git commit -m "feat(ref_repo): add $d"
done

echo "完成。最近提交："
git log --oneline -30
