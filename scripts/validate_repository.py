from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024
FORBIDDEN_EXTS = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".zip", ".rar", ".7z", ".pdf", ".docx", ".pptx"}
FORBIDDEN_PARTS = {"datasets", "runs", "weights", "checkpoints", "wandb"}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
REQUIRED = {
    "README.md", "AGENTS.md", "coordination/PROJECT_STATE.md", "coordination/DECISIONS.md",
    "coordination/CURRENT_ROUND.md", "coordination/REPORT_TEMPLATE.md", "data/DATA_MANIFEST.csv",
    "models/MODEL_MANIFEST.csv",
}
REPORT_HEADINGS = {
    "## 任务", "## 实际完成", "## 证据与来源", "## 修改文件", "## 验证命令与结果",
    "## 未完成或阻塞", "## 对下游的影响", "## 下一步建议",
}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def full_validation():
    errors = []
    present = {relative(p) for p in ROOT.rglob("*") if p.is_file() and not any(part in SKIP_PARTS for part in p.parts)}
    for name in sorted(REQUIRED - present):
        errors.append(f"缺少关键文件：{name}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        rel = relative(path)
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"文件超过20 MiB：{rel}")
        if path.suffix.lower() in FORBIDDEN_EXTS:
            errors.append(f"禁止扩展名：{rel}")
        if any(part.lower() in FORBIDDEN_PARTS for part in path.parts):
            errors.append(f"禁止目录：{rel}")
        if rel.startswith("reports/") and path.suffix.lower() == ".md" and path.name.lower() != "readme.md":
            text = path.read_text(encoding="utf-8")
            for heading in sorted(REPORT_HEADINGS):
                if heading not in text:
                    errors.append(f"报告缺少章节 {heading}：{rel}")
    return errors


def changed_validation(list_file):
    changed = [line.strip().replace("\\", "/") for line in Path(list_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    meaningful = [p for p in changed if not p.startswith(("reports/", "coordination/", ".github/")) and p not in {"README.md", "AGENTS.md"}]
    reports = [p for p in changed if p.startswith("reports/") and p.endswith(".md")]
    return ["PR修改了研究、代码或数据索引，但没有新增/更新独立报告文件。"] if meaningful and not reports else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed-files")
    args = parser.parse_args()
    errors = full_validation()
    if args.changed_files:
        errors.extend(changed_validation(args.changed_files))
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors))
        sys.exit(1)
    print("仓库校验通过。")


if __name__ == "__main__":
    main()
