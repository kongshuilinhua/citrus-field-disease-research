"""为一次复现实验生成可核验的截图归档。

该脚本不修改训练结果，只从运行目录中读取日志和原生图片，生成：
1. 每项实验必有的状态摘要图；
2. YOLO 原生结果图、混淆矩阵和预测样例的副本；
3. SimCLR / DANN 的训练曲线；
4. Leaf-Only-SAM 的输出图片副本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


EXPERIMENTS = {
    "leafonly": "[53] Leaf-Only-SAM 六张样例",
    "simclr": "[55] 无泄漏 SimCLR",
    "cms28": "[28] CMS 低保真重建分类",
    "scidb_baseline": "[36] SciDB YOLOv8n 检测基线",
    "rebuild36": "[36] 三通道低保真重建检测",
    "dann": "[35] DANN 重跑并保存权重",
    "official36": "[36] 作者补充代码，YOLOv8s 规模",
}


def result_roots(run_root: Path, experiment: str) -> list[Path]:
    return {
        "leafonly": [run_root / "leafonlysam"],
        "simclr": [run_root / "simclr"],
        "cms28": [
            run_root / "yolov8cls" / "citrusuat_cms",
            run_root / "yolov8cls" / "citrusuat_cms_val",
        ],
        "scidb_baseline": [
            run_root / "scidb_det" / "yolov8n",
            run_root / "scidb_det" / "yolov8n_val",
            run_root / "scidb_det" / "yolov8n_test",
        ],
        "rebuild36": [
            run_root / "scidb_det" / "yolov8n_3ch",
            run_root / "scidb_det" / "yolov8n_3ch_val",
            run_root / "scidb_det" / "yolov8n_3ch_test",
        ],
        "official36": [
            run_root / "scidb_det" / "powerneck_c2f_original",
            run_root / "scidb_det" / "powerneck_c2f_original_val",
            run_root / "scidb_det" / "powerneck_c2f_original_test",
        ],
        "dann": [run_root / "dann"],
    }.get(experiment, [])


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def tail_text(path: Path, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()[-max_lines:]


def select_key_lines(lines: list[str], limit: int = 13) -> list[str]:
    patterns = (
        "top1",
        "top5",
        "map50",
        "strict",
        "严格",
        "2类",
        "loss",
        "best",
        "saved",
        "complete",
        "error",
        "exception",
        "traceback",
        "cuda out of memory",
        "no such file",
    )
    cleaned = [
        re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
        for line in lines
    ]
    cleaned = [
        line
        for line in cleaned
        if line and not re.search(r"(it/s|s/it|\d+%\|)", line, re.I)
    ]
    selected = [line for line in cleaned if any(p in line.lower() for p in patterns)]
    if not selected:
        selected = cleaned[-limit:]
    return selected[-limit:]


def artifact_summary(run_root: Path, experiment: str) -> list[str]:
    roots = result_roots(run_root, experiment)
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    suffix_counts: dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower() or "[无扩展名]"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    detail = "、".join(f"{suffix}={count}" for suffix, count in sorted(suffix_counts.items()))
    return [f"原生产物文件：{len(files)} 个", f"类型统计：{detail or '暂无'}"]


def make_summary(
    run_root: Path,
    experiment: str,
    exit_code: int,
    output: Path,
    screenshot_count: int,
) -> None:
    stdout = run_root / "logs" / f"{experiment}.stdout.log"
    stderr = run_root / "logs" / f"{experiment}.stderr.log"
    stdout_lines = tail_text(stdout)
    stderr_lines = tail_text(stderr)
    key_lines = select_key_lines(stdout_lines + stderr_lines)

    status = "成功" if exit_code == 0 else "失败"
    status_color = (25, 135, 84) if exit_code == 0 else (190, 44, 54)
    title_font = font(34, bold=True)
    heading_font = font(23, bold=True)
    body_font = font(20)
    small_font = font(17)

    wrapped: list[str] = []
    for line in key_lines:
        wrapped.extend(textwrap.wrap(line, width=88, replace_whitespace=False) or [""])
    wrapped = wrapped[-18:]

    height = max(760, 550 + len(wrapped) * 29)
    image = Image.new("RGB", (1500, height), (247, 249, 252))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, 1500, 104), fill=(30, 42, 62))
    draw.text((48, 28), "复现实验结果留存", fill="white", font=title_font)
    draw.rounded_rectangle((1240, 25, 1448, 78), radius=18, fill=status_color)
    draw.text((1300, 35), status, fill="white", font=heading_font)

    y = 137
    draw.text((48, y), EXPERIMENTS.get(experiment, experiment), fill=(25, 32, 45), font=heading_font)
    y += 47
    draw.text((48, y), f"运行 ID：{run_root.name}", fill=(55, 65, 81), font=body_font)
    y += 34
    draw.text((48, y), f"退出码：{exit_code}    截图归档：{screenshot_count} 张（摘要图除外）", fill=(55, 65, 81), font=body_font)
    y += 48

    draw.line((48, y, 1450, y), fill=(205, 211, 220), width=2)
    y += 26
    draw.text((48, y), "产物概览", fill=(25, 32, 45), font=heading_font)
    y += 39
    for line in artifact_summary(run_root, experiment):
        draw.text((70, y), f"• {line}", fill=(55, 65, 81), font=body_font)
        y += 32

    y += 13
    draw.text((48, y), "日志关键行", fill=(25, 32, 45), font=heading_font)
    y += 40
    if wrapped:
        for line in wrapped:
            draw.text((70, y), line, fill=(45, 55, 70), font=small_font)
            y += 29
    else:
        draw.text((70, y), "暂无可提取日志；请查看原始 stdout/stderr 文件。", fill=(120, 67, 35), font=body_font)
        y += 32

    draw.text(
        (48, height - 52),
        "核验说明：本图由已落盘日志与产物自动生成；原始文件及 SHA-256 清单保留在同一运行目录。",
        fill=(92, 103, 120),
        font=small_font,
    )
    image.save(output)


def copy_native_images(run_root: Path, experiment: str, target: Path) -> int:
    source_roots = result_roots(run_root, experiment)
    patterns = (
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxPR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_pred.jpg",
        "*_leafonly.png",
    )
    copied = 0
    seen: set[str] = set()
    for source_root in source_roots:
        if not source_root.exists():
            continue
        for pattern in patterns:
            for source in source_root.rglob(pattern):
                relative = source.relative_to(source_root)
                safe_name = f"{source_root.name}__{'__'.join(relative.parts)}"
                if safe_name in seen:
                    continue
                seen.add(safe_name)
                shutil.copy2(source, target / safe_name)
                copied += 1
    return copied


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_experiment_hashes(run_root: Path, experiment: str, target: Path) -> None:
    candidates: set[Path] = set()
    for root in result_roots(run_root, experiment):
        if root.exists():
            candidates.update(path for path in root.rglob("*") if path.is_file())
    for suffix in ("stdout.log", "stderr.log", "screenshots.stdout.log", "screenshots.stderr.log"):
        path = run_root / "logs" / f"{experiment}.{suffix}"
        if path.exists():
            candidates.add(path)
    candidates.update(
        path
        for path in target.iterdir()
        if path.is_file() and path.name != "experiment_sha256.json"
    )
    entries = []
    for path in sorted(candidates):
        entries.append(
            {
                "path": str(path.relative_to(run_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (target / "experiment_sha256.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "file_count": len(entries),
                "files": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_simclr(run_root: Path, target: Path) -> int:
    lines = tail_text(run_root / "logs" / "simclr.stdout.log", max_lines=10000)
    points: list[tuple[int, float]] = []
    pattern = re.compile(r"ep\s*(\d+).*?(?:NT-Xent\s*)?loss\s*=\s*([0-9.]+)", re.I)
    for line in lines:
        match = pattern.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(2))))
    if not points:
        return 0
    plt.figure(figsize=(10, 5.6))
    plt.plot([x for x, _ in points], [y for _, y in points], color="#2457d6", linewidth=2)
    plt.scatter([x for x, _ in points], [y for _, y in points], color="#2457d6", s=18)
    plt.title("SimCLR 训练损失")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(target / "simclr_loss_curve.png", dpi=160)
    plt.close()
    return 1


def plot_dann(run_root: Path, target: Path) -> int:
    lines = tail_text(run_root / "logs" / "dann.stdout.log", max_lines=10000)
    pattern = re.compile(
        r"\[(source_only|dann)\]\s*ep\s*(\d+).*?严格5类\s*=\s*([0-9.]+)%.*?2类\s*=\s*([0-9.]+)%",
        re.I,
    )
    series: dict[str, list[tuple[int, float, float]]] = {"source_only": [], "dann": []}
    for line in lines:
        match = pattern.search(line)
        if match:
            series[match.group(1).lower()].append(
                (int(match.group(2)), float(match.group(3)), float(match.group(4)))
            )
    if not any(series.values()):
        return 0
    plt.figure(figsize=(10, 5.6))
    colors = {"source_only": "#697386", "dann": "#e85d04"}
    labels = {"source_only": "Source-only", "dann": "DANN"}
    for name, points in series.items():
        if not points:
            continue
        epochs = [point[0] for point in points]
        plt.plot(
            epochs,
            [point[1] for point in points],
            color=colors[name],
            linewidth=2,
            label=f"{labels[name]} 严格5类",
        )
        plt.plot(
            epochs,
            [point[2] for point in points],
            color=colors[name],
            linewidth=2,
            linestyle="--",
            label=f"{labels[name]} 受限2类",
        )
    plt.title("DANN 目标域准确率")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target / "dann_accuracy_curve.png", dpi=160)
    plt.close()
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    target = args.run_root / "screenshots" / args.experiment
    target.mkdir(parents=True, exist_ok=True)

    count = copy_native_images(args.run_root, args.experiment, target)
    if args.experiment == "simclr":
        count += plot_simclr(args.run_root, target)
    elif args.experiment == "dann":
        count += plot_dann(args.run_root, target)

    make_summary(
        args.run_root,
        args.experiment,
        args.exit_code,
        target / "00_实验状态摘要.png",
        count,
    )
    manifest = {
        "experiment": args.experiment,
        "exit_code": args.exit_code,
        "screenshot_count": count + 1,
        "files": sorted(
            path.name
            for path in target.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
    }
    (target / "screenshots_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_experiment_hashes(args.run_root, args.experiment, target)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
