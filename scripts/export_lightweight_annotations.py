from pathlib import Path
import argparse
import csv
from openpyxl import load_workbook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ws = load_workbook(args.source, read_only=True, data_only=True)["合并预标注"]
    headers = [c.value for c in ws[1]]
    keep = [
        "总序号", "数据源", "源内序号", "文件名（点击打开归档原图）", "拍摄设备", "主要对象",
        "可见症状预标签", "疑似解释（非确诊）", "可见程度", "画面结构", "清晰度", "疑似连拍组",
        "建议训练标注", "预标置信度", "专家复核", "建议处置", "备注", "分辨率", "文件大小(MB)",
        "拍摄时间", "SHA-256",
    ]
    index = {name: headers.index(name) for name in keep}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keep)
        for row in ws.iter_rows(min_row=2, values_only=True):
            writer.writerow([row[index[name]] for name in keep])
    print(f"rows={ws.max_row - 1} output={args.output}")


if __name__ == "__main__":
    main()

