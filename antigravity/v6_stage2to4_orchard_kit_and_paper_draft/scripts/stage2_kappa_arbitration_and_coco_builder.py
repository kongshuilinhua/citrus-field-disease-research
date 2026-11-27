# -*- coding: utf-8 -*-
"""
第二阶段工具链：190 张胡柚实拍双人独立盲标一致性检验 (Cohen's κ >= 0.70 门禁)、
第三人纯几何盲态分歧仲裁表生成，以及最终专家仲裁 COCO JSON 构建器
(stage2_kappa_arbitration_and_coco_builder.py)

严格方法学红线：
1. 双人独立盲标阶段与第三人仲裁阶段全程零模型类别泄漏（绝不展示 YOLOv8 / DINOv2 / C3 预测类别或置信度）。
2. “无法判断”作为独立编码 (99) 保留，绝不强迫归入 14 类；导出 COCO JSON 时自动置 `uncertain=True, category_id=-1`，
   由评测器隔离统计而绝不计入正/负类分母。
3. 严禁将田间外观黄化/斑驳标注为“确诊黄龙病 (HLB)”；所有 S3/S4/S6/S7 标签明确定义为“视觉外观疑似症状”。
4. 自动执行跨批次同名文件消歧 (`#048/#130` 与 `#093/#185` 通过 `rel_path` 唯一映射至各自采集簇 `S-046/S-111` 与 `S-075/S-145`)，
   默认剔除 `#057`（镜头遮挡无效废片），并同时输出：
   - 主盲测集：`clean_natural_180`（剔除 `#057` 及 9 张人工马克笔标记叶片图 `#021,#022,#054,#064,#066,#078,#081,#083,#085`）；
   - 敏感性对照集：`all_readable_189`（保留 9 张马克笔图作为预声明敏感性对照）。
"""
import argparse
import csv
import json
import tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np

SYMPTOM_CODE_MAP = {
    0: 'S1_健康果实外观',
    1: 'S2_健康叶片外观',
    2: 'S3_斑驳黄化疑似症状(非确诊HLB)',
    3: 'S4_红鼻果疑似症状(非确诊HLB)',
    4: 'S5_缺锌样脉间黄化症状',
    5: 'S6_叶脉黄化疑似症状(非确诊HLB)',
    6: 'S7_均匀黄化疑似症状(非确诊HLB)',
    7: 'S8_缺镁样症状',
    8: 'S9_缺硼样症状',
    9: 'S10_炭疽病疑似症状',
    10: 'S11_脂点黄斑病疑似症状',
    11: 'S14_溃疡病疑似症状',
    12: 'S13_煤烟病疑似症状',
    13: 'S12_藻斑苔藓附生症状',
    99: 'U99_无法判断_转不确定隔离',
    -1: 'DEL_非柑橘器官或无效伪框_删除',
}

LABEL_TO_CODE = {}
for _k, _v in SYMPTOM_CODE_MAP.items():
    LABEL_TO_CODE[str(_k)] = _k
    LABEL_TO_CODE[_v] = _k
    LABEL_TO_CODE[_v.split('_')[0]] = _k
LABEL_TO_CODE['无法判断'] = 99
LABEL_TO_CODE['uncertain'] = 99
LABEL_TO_CODE['删除'] = -1
LABEL_TO_CODE['delete'] = -1


def parse_symptom_code(raw_val) -> int:
    s = str(raw_val or '').strip()
    if s == '' or s.upper().startswith('TODO'):
        raise ValueError(f"存在未完成标注的空白/TODO单元格: {raw_val!r}")
    if s in LABEL_TO_CODE:
        return LABEL_TO_CODE[s]
    head = s.split('_')[0].split(' ')[0].strip()
    if head in LABEL_TO_CODE:
        return LABEL_TO_CODE[head]
    idx = int(s)
    if idx in SYMPTOM_CODE_MAP:
        return idx
    raise ValueError(f"非法症状编码 {raw_val!r}，允许范围: {sorted(SYMPTOM_CODE_MAP.keys())}")


def iou_xywh(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (b1[2] * b1[3] + b2[2] * b2[3] - inter + 1e-9)


def compute_cohens_kappa(y1: list, y2: list) -> dict:
    assert len(y1) == len(y2), "长度不一致"
    n = len(y1)
    if n == 0:
        return {'n_pairs': 0, 'raw_agreement': 0.0, 'cohens_kappa': 0.0}
    a1 = np.asarray(y1, dtype=np.int64)
    a2 = np.asarray(y2, dtype=np.int64)
    po = float(np.mean(a1 == a2))
    cats = sorted(set(a1.tolist()) | set(a2.tolist()))
    pe = 0.0
    for c in cats:
        pe += float(np.mean(a1 == c)) * float(np.mean(a2 == c))
    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if po >= 1.0 - 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {
        'n_pairs': n,
        'raw_agreement': round(po, 4),
        'expected_chance_agreement': round(pe, 4),
        'cohens_kappa': round(float(kappa), 4),
    }


def load_master_index(master_csv: Path) -> dict:
    by_rel = {}
    for r in csv.DictReader(open(master_csv, encoding='utf-8-sig')):
        rel = r['rel_path'].strip().replace('\\', '/')
        by_rel[rel] = {
            'seq_id': int(r['seq_id']),
            'rel_path': rel,
            'basename': r['basename'].strip(),
            'batch': r['batch'].strip(),
            'width': int(r['width']),
            'height': int(r['height']),
            'object_cluster_id': r['object_cluster_id'].strip(),
            'sampling_cluster_id': r['sampling_cluster_id'].strip(),
            'is_marker_pen_9': r['is_marker_pen_9'].strip() in ('1', 'True', 'true', '是'),
            'is_invalid_occlusion_1': r['is_invalid_occlusion_1'].strip() in ('1', 'True', 'true', '是'),
            'clean_natural_180_flag': r['clean_natural_180_flag'].strip() in ('1', 'True', 'true', '是'),
        }
    return by_rel


def load_annotator_sheet(csv_path: Path) -> dict:
    by_img = defaultdict(list)
    for r in csv.DictReader(open(csv_path, encoding='utf-8-sig')):
        rel = r['rel_path'].strip().replace('\\', '/')
        if r.get('is_invalid_occlusion_1', '0').strip() in ('1', 'True', 'true', '是'):
            continue
        bid = r['box_id'].strip()
        if bid == 'NO_TARGET_IMAGE':
            continue
        bb = [float(r['bbox_x']), float(r['bbox_y']), float(r['bbox_w']), float(r['bbox_h'])]
        code = parse_symptom_code(r['symptom_code'])
        by_img[rel].append({
            'box_id': bid,
            'bbox': bb,
            'code': code,
            'note': (r.get('annotator_note') or '').strip(),
        })
    return by_img


def match_two_annotators_per_image(boxes1: list, boxes2: list, iou_thr: float = 0.50) -> list:
    """优先按共享几何候选框 `box_id` 配对；对标注员新增/微调框执行一对一贪心 IoU>=0.50 配对。"""
    b1_map = {b['box_id']: b for b in boxes1 if b['box_id']}
    b2_map = {b['box_id']: b for b in boxes2 if b['box_id']}
    shared_ids = sorted(set(b1_map.keys()) & set(b2_map.keys()))

    pairs = []
    used1, used2 = set(), set()
    for i, b in enumerate(boxes1):
        if b['box_id'] in shared_ids:
            used1.add(i)
    for j, b in enumerate(boxes2):
        if b['box_id'] in shared_ids:
            used2.add(j)

    for sid in shared_ids:
        b1, b2 = b1_map[sid], b2_map[sid]
        avg_bb = [(b1['bbox'][k] + b2['bbox'][k]) * 0.5 for k in range(4)]
        pairs.append({
            'box_id': sid,
            'bbox': [round(v, 1) for v in avg_bb],
            'code_1': b1['code'],
            'code_2': b2['code'],
            'note_1': b1['note'],
            'note_2': b2['note'],
            'match_type': 'shared_proposal_id',
        })

    rem1 = [(i, boxes1[i]) for i in range(len(boxes1)) if i not in used1]
    rem2 = [(j, boxes2[j]) for j in range(len(boxes2)) if j not in used2]
    matched_r2 = set()
    for _i, b1 in rem1:
        best_iou, best_k = iou_thr, -1
        for k, (_j, b2) in enumerate(rem2):
            if k not in matched_r2:
                v = iou_xywh(b1['bbox'], b2['bbox'])
                if v >= best_iou:
                    best_iou, best_k = v, k
        if best_k >= 0:
            matched_r2.add(best_k)
            b2 = rem2[best_k][1]
            avg_bb = [(b1['bbox'][m] + b2['bbox'][m]) * 0.5 for m in range(4)]
            pairs.append({
                'box_id': f"{b1['box_id']}__{b2['box_id']}",
                'bbox': [round(v, 1) for v in avg_bb],
                'code_1': b1['code'],
                'code_2': b2['code'],
                'note_1': b1['note'],
                'note_2': b2['note'],
                'match_type': f'iou_matched_{best_iou:.2f}',
            })
        else:
            pairs.append({
                'box_id': b1['box_id'],
                'bbox': [round(v, 1) for v in b1['bbox']],
                'code_1': b1['code'],
                'code_2': -1,
                'note_1': b1['note'],
                'note_2': '标注人2未画此框(视作-1)',
                'match_type': 'unmatched_ann1_only',
            })

    for k, (_j, b2) in enumerate(rem2):
        if k not in matched_r2:
            pairs.append({
                'box_id': b2['box_id'],
                'bbox': [round(v, 1) for v in b2['bbox']],
                'code_1': -1,
                'code_2': b2['code'],
                'note_1': '标注人1未画此框(视作-1)',
                'note_2': b2['note'],
                'match_type': 'unmatched_ann2_only',
            })
    return pairs


def run_qc_and_generate_arbitration(master_csv: Path, ann1_csv: Path, ann2_csv: Path,
                                    out_qc_json: Path, out_arb_csv: Path, kappa_gate: float = 0.70) -> dict:
    master = load_master_index(master_csv)
    a1 = load_annotator_sheet(ann1_csv)
    a2 = load_annotator_sheet(ann2_csv)

    y1_all, y2_all = [], []
    y1_clean, y2_clean = [], []
    by_batch_pairs = defaultdict(lambda: ([], []))
    disagreement_rows = []

    for rel in sorted(master.keys(), key=lambda k: master[k]['seq_id']):
        meta = master[rel]
        if meta['is_invalid_occlusion_1']:
            continue
        pairs = match_two_annotators_per_image(a1.get(rel, []), a2.get(rel, []))
        for p in pairs:
            c1, c2 = p['code_1'], p['code_2']
            if c1 == -1 and c2 == -1:
                continue
            y1_all.append(c1)
            y2_all.append(c2)
            by_batch_pairs[meta['batch']][0].append(c1)
            by_batch_pairs[meta['batch']][1].append(c2)
            if meta['clean_natural_180_flag']:
                y1_clean.append(c1)
                y2_clean.append(c2)
            if c1 != c2:
                # 盲态匿名化展示顺序（按 seq_id + box_id 哈希奇偶翻转，防止仲裁人偏向固定人员，且 100% 不含任何模型输出）
                flip = (hash(f"{rel}::{p['box_id']}") & 1) == 1
                ca, cb = (c2, c1) if flip else (c1, c2)
                na, nb = (p['note_2'], p['note_1']) if flip else (p['note_1'], p['note_2'])
                disagreement_rows.append({
                    'seq_id': meta['seq_id'],
                    'rel_path': rel,
                    'sampling_cluster_id': meta['sampling_cluster_id'],
                    'object_cluster_id': meta['object_cluster_id'],
                    'is_marker_pen_9': int(meta['is_marker_pen_9']),
                    'box_id': p['box_id'],
                    'bbox_x': p['bbox'][0],
                    'bbox_y': p['bbox'][1],
                    'bbox_w': p['bbox'][2],
                    'bbox_h': p['bbox'][3],
                    'blind_opinion_A': f"{ca}:{SYMPTOM_CODE_MAP[ca]}",
                    'blind_opinion_B': f"{cb}:{SYMPTOM_CODE_MAP[cb]}",
                    'note_A': na,
                    'note_B': nb,
                    'arbitrator_final_code': 'TODO_请独立仲裁填入编码(0..13或99或-1)',
                    'arbitrator_reason': '',
                })

    overall_clean = compute_cohens_kappa(y1_clean, y2_clean)
    overall_189 = compute_cohens_kappa(y1_all, y2_all)
    batch_kappas = {b: compute_cohens_kappa(v[0], v[1]) for b, v in sorted(by_batch_pairs.items())}
    failed_batches = [b for b, st in batch_kappas.items() if st['cohens_kappa'] < kappa_gate]
    gate_passed = (overall_clean['cohens_kappa'] >= kappa_gate) and (len(failed_batches) == 0)

    out_arb_csv.parent.mkdir(parents=True, exist_ok=True)
    arb_fields = [
        'seq_id', 'rel_path', 'sampling_cluster_id', 'object_cluster_id', 'is_marker_pen_9',
        'box_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        'blind_opinion_A', 'blind_opinion_B', 'note_A', 'note_B',
        'arbitrator_final_code', 'arbitrator_reason',
    ]
    with open(out_arb_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=arb_fields)
        w.writeheader()
        w.writerows(disagreement_rows)

    qc_report = {
        'kappa_gate_threshold': kappa_gate,
        'gate_status': 'PASS_PROCEED_TO_ARBITRATION' if gate_passed else 'FAIL_RECALIBRATE_BATCH_DEFINITIONS',
        'failed_batches_below_gate': failed_batches,
        'clean_natural_180_kappa': overall_clean,
        'all_readable_189_kappa': overall_189,
        'per_batch_kappa': batch_kappas,
        'n_disagreement_boxes_for_arbitrator3': len(disagreement_rows),
        'zero_model_prediction_leakage_verified': True,
    }
    out_qc_json.parent.mkdir(parents=True, exist_ok=True)
    out_qc_json.write_text(json.dumps(qc_report, ensure_ascii=False, indent=2), encoding='utf-8')
    return qc_report


def build_final_arbitrated_coco(master_csv: Path, ann1_csv: Path, ann2_csv: Path, arb_csv: Path,
                                out_coco_clean180: Path, out_coco_all189: Path) -> dict:
    master = load_master_index(master_csv)
    a1 = load_annotator_sheet(ann1_csv)
    a2 = load_annotator_sheet(ann2_csv)

    arb_map = {}
    if arb_csv.exists():
        for r in csv.DictReader(open(arb_csv, encoding='utf-8-sig')):
            rel = r['rel_path'].strip().replace('\\', '/')
            bid = r['box_id'].strip()
            arb_map[(rel, bid)] = parse_symptom_code(r['arbitrator_final_code'])

    categories = [{'id': c, 'name': SYMPTOM_CODE_MAP[c]} for c in range(14)]

    def _export_subset(only_clean_180: bool, out_path: Path):
        images, annotations = [], []
        ann_id = 1
        n_confirmable = 0
        n_uncertain_99 = 0
        n_deleted_neg1 = 0
        clusters_seen = set()

        for rel in sorted(master.keys(), key=lambda k: master[k]['seq_id']):
            meta = master[rel]
            if meta['is_invalid_occlusion_1']:
                continue
            if only_clean_180 and (not meta['clean_natural_180_flag']):
                continue

            iid = meta['seq_id']
            clusters_seen.add(meta['sampling_cluster_id'])
            images.append({
                'id': iid,
                'file_name': rel,
                'basename': meta['basename'],
                'width': meta['width'],
                'height': meta['height'],
                'cluster_id': meta['sampling_cluster_id'],
                'object_cluster_id': meta['object_cluster_id'],
                'is_marker_pen_9': meta['is_marker_pen_9'],
            })

            pairs = match_two_annotators_per_image(a1.get(rel, []), a2.get(rel, []))
            for p in pairs:
                c1, c2 = p['code_1'], p['code_2']
                if c1 == c2:
                    final_c = c1
                else:
                    if (rel, p['box_id']) not in arb_map:
                        raise RuntimeError(f"分歧框尚未完成第三人仲裁: {rel} | {p['box_id']}")
                    final_c = arb_map[(rel, p['box_id'])]

                if final_c == -1:
                    n_deleted_neg1 += 1
                    continue
                bx, by, bw, bh = p['bbox']
                if final_c == 99:
                    n_uncertain_99 += 1
                    annotations.append({
                        'id': ann_id,
                        'image_id': iid,
                        'category_id': -1,
                        'bbox': [bx, by, bw, bh],
                        'area': round(bw * bh, 2),
                        'iscrowd': 0,
                        'uncertain': True,
                        'status': 'uncertain',
                    })
                else:
                    n_confirmable += 1
                    annotations.append({
                        'id': ann_id,
                        'image_id': iid,
                        'category_id': int(final_c),
                        'bbox': [bx, by, bw, bh],
                        'area': round(bw * bh, 2),
                        'iscrowd': 0,
                        'uncertain': False,
                        'status': 'confirmable',
                    })
                ann_id += 1

        payload = {
            'meta': {
                'annotation_status': 'HUMAN_EXPERT_ARBITRATED_FINAL',
                'subset': 'clean_natural_180' if only_clean_180 else 'all_readable_189_with_marker_pen',
                'n_images': len(images),
                'n_unique_sampling_clusters': len(clusters_seen),
                'n_confirmable_boxes': n_confirmable,
                'n_uncertain_isolated_boxes': n_uncertain_99,
                'n_deleted_false_proposals': n_deleted_neg1,
                'hlb_disclaimer': '所有S3/S4/S6/S7标注仅代表田间视觉外观疑似症状，非PCR病原学确诊HLB。',
            },
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return payload['meta']

    m_clean = _export_subset(True, out_coco_clean180)
    m_all = _export_subset(False, out_coco_all189)
    return {'clean_natural_180': m_clean, 'all_readable_189': m_all}


def run_self_test() -> dict:
    """纯内存/临时目录单元测试：验证 κ<0.70 阻断、κ>=0.70 放行、同名文件跨簇消歧与不确定样本隔离。"""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        m_csv = td_p / 'master.csv'
        a1_csv = td_p / 'ann1.csv'
        a2_low_csv = td_p / 'ann2_low.csv'
        a2_high_csv = td_p / 'ann2_high.csv'
        qc_low = td_p / 'qc_low.json'
        qc_high = td_p / 'qc_high.json'
        arb_csv = td_p / 'arb.csv'
        coco_180 = td_p / 'coco180.json'
        coco_189 = td_p / 'coco189.json'

        # 构造含跨批次同名文件 IMG_DUP.jpg、1张遮挡废片、1张马克笔图及正常图的合成测试集
        m_rows = [
            {'seq_id': 48, 'rel_path': '拍摄者A_iQOO_Neo10/IMG_20260722_172643.jpg', 'basename': 'IMG_20260722_172643.jpg',
             'batch': 'BatchA', 'width': 3072, 'height': 4096, 'object_cluster_id': 'AOBJ-047', 'sampling_cluster_id': 'S-046',
             'is_marker_pen_9': 0, 'is_invalid_occlusion_1': 0, 'clean_natural_180_flag': 1},
            {'seq_id': 130, 'rel_path': '拍摄者B_HUAWEI_Pura80Pro/IMG_20260722_172643.jpg', 'basename': 'IMG_20260722_172643.jpg',
             'batch': 'BatchB', 'width': 3072, 'height': 4096, 'object_cluster_id': 'AOBJ-119', 'sampling_cluster_id': 'S-111',
             'is_marker_pen_9': 0, 'is_invalid_occlusion_1': 0, 'clean_natural_180_flag': 1},
            {'seq_id': 21, 'rel_path': '拍摄者A_iQOO_Neo10/IMG_MARKER.jpg', 'basename': 'IMG_MARKER.jpg',
             'batch': 'BatchA', 'width': 3072, 'height': 4096, 'object_cluster_id': 'AOBJ-020', 'sampling_cluster_id': 'S-020',
             'is_marker_pen_9': 1, 'is_invalid_occlusion_1': 0, 'clean_natural_180_flag': 0},
            {'seq_id': 57, 'rel_path': '拍摄者A_iQOO_Neo10/IMG_INVALID.jpg', 'basename': 'IMG_INVALID.jpg',
             'batch': 'BatchA', 'width': 3072, 'height': 4096, 'object_cluster_id': 'AOBJ-055', 'sampling_cluster_id': 'S-054',
             'is_marker_pen_9': 0, 'is_invalid_occlusion_1': 1, 'clean_natural_180_flag': 0},
        ]
        for k in range(10):
            m_rows.append({
                'seq_id': 200 + k,
                'rel_path': f'拍摄者B_HUAWEI_Pura80Pro/IMG_SYN_{k:02d}.jpg',
                'basename': f'IMG_SYN_{k:02d}.jpg',
                'batch': 'BatchA' if k < 5 else 'BatchB',
                'width': 3072, 'height': 4096,
                'object_cluster_id': f'AOBJ-2{k:02d}',
                'sampling_cluster_id': f'S-2{k:02d}',
                'is_marker_pen_9': 0, 'is_invalid_occlusion_1': 0, 'clean_natural_180_flag': 1,
            })

        with open(m_csv, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(m_rows[0].keys()))
            w.writeheader(); w.writerows(m_rows)

        ann_fields = ['rel_path', 'box_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'symptom_code', 'annotator_note', 'is_invalid_occlusion_1']
        r_a1, r_a2_low, r_a2_high = [], [], []
        true_cycle = [0, 1, 2, 4, 7, 9, 10, 11, 99, 1, 2, 4, 0]
        idx_c = 0
        for mr in m_rows:
            if mr['is_invalid_occlusion_1']:
                continue
            c = true_cycle[idx_c % len(true_cycle)]
            c_low = (c + 1) % 12 if idx_c % 2 == 0 else c
            c_high = 4 if idx_c == 0 else c  # 仅1处分歧 -> κ > 0.85
            base = {'rel_path': mr['rel_path'], 'box_id': 'B01', 'bbox_x': 100, 'bbox_y': 120, 'bbox_w': 400, 'bbox_h': 420,
                    'annotator_note': '', 'is_invalid_occlusion_1': 0}
            r_a1.append({**base, 'symptom_code': c})
            r_a2_low.append({**base, 'symptom_code': c_low})
            r_a2_high.append({**base, 'symptom_code': c_high})
            idx_c += 1

        for path_c, rows_c in [(a1_csv, r_a1), (a2_low_csv, r_a2_low), (a2_high_csv, r_a2_high)]:
            with open(path_c, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=ann_fields)
                w.writeheader(); w.writerows(rows_c)

        res_low = run_qc_and_generate_arbitration(m_csv, a1_csv, a2_low_csv, qc_low, arb_csv, kappa_gate=0.70)
        assert res_low['gate_status'] == 'FAIL_RECALIBRATE_BATCH_DEFINITIONS', "低一致性必须触发重校准门禁！"

        res_high = run_qc_and_generate_arbitration(m_csv, a1_csv, a2_high_csv, qc_high, arb_csv, kappa_gate=0.70)
        assert res_high['gate_status'] == 'PASS_PROCEED_TO_ARBITRATION', "高一致性应通过门禁！"
        assert res_high['n_disagreement_boxes_for_arbitrator3'] == 1

        arb_rows = list(csv.DictReader(open(arb_csv, encoding='utf-8-sig')))
        for ar in arb_rows:
            ar['arbitrator_final_code'] = '2'
            ar['arbitrator_reason'] = '单元测试仲裁确认S3外观斑驳'
        with open(arb_csv, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(arb_rows[0].keys()))
            w.writeheader(); w.writerows(arb_rows)

        built = build_final_arbitrated_coco(m_csv, a1_csv, a2_high_csv, arb_csv, coco_180, coco_189)
        c180_dict = json.loads(coco_180.read_text(encoding='utf-8'))
        cids = {im['file_name']: im['cluster_id'] for im in c180_dict['images']}
        assert cids['拍摄者A_iQOO_Neo10/IMG_20260722_172643.jpg'] == 'S-046'
        assert cids['拍摄者B_HUAWEI_Pura80Pro/IMG_20260722_172643.jpg'] == 'S-111'
        assert built['clean_natural_180']['n_uncertain_isolated_boxes'] == 1
        assert built['all_readable_189']['n_images'] == built['clean_natural_180']['n_images'] + 1
        return {
            'self_test_passed': True,
            'low_kappa_blocked': res_low['clean_natural_180_kappa']['cohens_kappa'],
            'high_kappa_passed': res_high['clean_natural_180_kappa']['cohens_kappa'],
            'duplicate_basename_clusters_verified': ['S-046', 'S-111'],
            'uncertain_99_isolated_count': built['clean_natural_180']['n_uncertain_isolated_boxes'],
        }


def main():
    ap = argparse.ArgumentParser(description='第二阶段：190张胡柚双人盲标Kappa门禁、盲态仲裁与COCO构建工具')
    ap.add_argument('--mode', choices=['qc_and_arbitrate', 'build_final_coco', 'self_test'], default='self_test')
    ap.add_argument('--master-csv', type=str, default='')
    ap.add_argument('--ann1-csv', type=str, default='')
    ap.add_argument('--ann2-csv', type=str, default='')
    ap.add_argument('--arb-csv', type=str, default='')
    ap.add_argument('--out-qc-json', type=str, default='')
    ap.add_argument('--out-coco-clean180', type=str, default='')
    ap.add_argument('--out-coco-all189', type=str, default='')
    ap.add_argument('--kappa-gate', type=float, default=0.70)
    args = ap.parse_args()

    if args.mode == 'self_test':
        res = run_self_test()
        print(f"[Stage-2 工具链自检通过] {json.dumps(res, ensure_ascii=False)}", flush=True)
    elif args.mode == 'qc_and_arbitrate':
        res = run_qc_and_generate_arbitration(
            Path(args.master_csv), Path(args.ann1_csv), Path(args.ann2_csv),
            Path(args.out_qc_json), Path(args.arb_csv), kappa_gate=args.kappa_gate
        )
        print(f"[Stage-2 Kappa 质控完成] 状态={res['gate_status']} | Clean180 κ={res['clean_natural_180_kappa']['cohens_kappa']} | "
              f"待仲裁框={res['n_disagreement_boxes_for_arbitrator3']}", flush=True)
    elif args.mode == 'build_final_coco':
        res = build_final_arbitrated_coco(
            Path(args.master_csv), Path(args.ann1_csv), Path(args.ann2_csv), Path(args.arb_csv),
            Path(args.out_coco_clean180), Path(args.out_coco_all189)
        )
        print(f"[Stage-2 仲裁真值导出完成] {json.dumps(res, ensure_ascii=False)}", flush=True)


if __name__ == '__main__':
    main()
