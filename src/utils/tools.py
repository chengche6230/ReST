import os
import motmetrics as mm
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

    accs = []
    names = []
    for c in range(cfg.DATASET.CAMS):
        gt_file = f'{cfg.DATASET.DIR}{cfg.DATASET.NAME}/{cfg.DATASET.SEQUENCE[0]}/output/gt_MOT/c{c}.txt'
        ts_file = os.path.join(output_dir, f'c{c}.txt')
        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")
        names.append(os.path.splitext(os.path.basename(ts_file))[0])
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5))
    summary = mh.compute_many(accs, metrics=metrics, generate_overall=True)#, name=names)
    logger.info(f'\n{mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)}')
