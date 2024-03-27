import os
import numpy as np
import motmetrics as mm
import pandas as pd
from loguru import logger

def udf_collate_fn(batch):
    return batch

def get_color(idx: int):
    idx = idx * 3
    return (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def evaluate(cfg, output_dir):
    metrics = list(mm.metrics.motchallenge_metrics)
    mh = mm.metrics.create()

    gt_dfs, ts_dfs = [], []
    for c in range(cfg.DATASET.CAMS):
        gt_file = f'{cfg.DATASET.DIR}{cfg.DATASET.NAME}/{cfg.DATASET.SEQUENCE[0]}/output/gt_MOT/c{c}.txt'
        ts_file = os.path.join(output_dir, f'c{c}.txt')
        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1).reset_index()
        ts = mm.io.loadtxt(ts_file, fmt="mot15-2D").reset_index()
        gt_dfs.append(gt)
        ts_dfs.append(ts)

    count_frames = 0
    for j, (gt, ts) in enumerate(zip(gt_dfs, ts_dfs)):
        gt["FrameId"] += count_frames
        ts["FrameId"] += count_frames
        count_frames += gt["FrameId"].max() + 1

    # stack gts and tss dataframes
    gt = pd.concat(gt_dfs, axis=0).set_index(['FrameId', 'Id'])
    ts = pd.concat(ts_dfs, axis=0).set_index(['FrameId', 'Id'])

    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    summary = mh.compute(acc, metrics=metrics)
    logger.info(f'\n{mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)}')

    if cfg.DATASET.NAME == 'Wildtrack':
        mh = mm.metrics.create()

        print('=========== Wildtrack GROUND PLANE evaluation ===========')
        gt_file = f'{cfg.DATASET.DIR}{cfg.DATASET.NAME}/{cfg.DATASET.SEQUENCE[0]}/output/gt_MOT/gp.txt'
        ts_file = os.path.join(output_dir, f'gp.txt')

        gt = np.loadtxt(gt_file, delimiter=',')
        t = np.loadtxt(ts_file, delimiter=',')

        acc = mm.MOTAccumulator(auto_id=True)
        for frame in np.unique(gt[:, 0]).astype(int):
            gt_dets = gt[gt[:, 0] == frame][:, (1, 6, 7)]
            t_dets = t[t[:, 0] == frame][:, (1, 7, 8)]

            # world grid to world coord in meters
            t_trans = (t_dets[:, 1:3] / 2.5) + np.array([360, 120])

            C = mm.distances.norm2squared_matrix(gt_dets[:, 1:3]  * 0.025, t_trans * 0.025, max_d2=1)
            C = np.sqrt(C)

            acc.update(gt_dets[:, 0].astype('int').tolist(), t_dets[:, 0].astype('int').tolist(), C)

        summary = mh.compute(acc, metrics=metrics)
        logger.info(
            f'\n{mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)}')
