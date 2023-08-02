import time
import os.path as osp
from loguru import logger

import dgl
from torchreid.utils import FeatureExtractor
from .models import NodeFeatureEncoder, EdgeFeatureEncoder, EdgePredictor, MPN#, FeatureExtractor
from .datasets.dataset import BaseGraphDataset
from .utils.tools import *
from .utils.tracklet import Tracklet

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler


class Tracker:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device=self.cfg.MODEL.DEVICE
        self.min_loss = 1e8

        if self.cfg.FE.CHOICE == 'CNN':
            # Default ReID model, OS-Net here
            self.feature_extractor = FeatureExtractor(
                model_name='osnet_ain_x1_0',
                model_path='logs/osnet_ain_ms_d_c.pth.tar',
                device=self.device
            )

        self.node_feature_encoder = NodeFeatureEncoder(self.cfg)
        self.edge_feature_encoder = EdgeFeatureEncoder(self.cfg)
        self.mpn = MPN(self.cfg)
        self.predictor = EdgePredictor(self.cfg)
        make_dir(self.cfg.OUTPUT.CKPT_DIR)

        if self.cfg.MODEL.MODE == 'test':
            self.output_dir = osp.join(self.cfg.OUTPUT.INFERENCE_DIR, f'test-{self.cfg.DATASET.NAME}-{self.cfg.DATASET.SEQUENCE[0]}-{int(time.time())}')
            self.tracklet = Tracklet(self.cfg, self.output_dir)
        else:
            self.output_dir = osp.join(self.cfg.OUTPUT.CKPT_DIR, f'train-{self.cfg.DATASET.NAME}-{self.cfg.DATASET.SEQUENCE[0]}-{self.cfg.SOLVER.TYPE}-{int(time.time())}')

        make_dir(self.output_dir)
        logger.add(f'{self.output_dir}/log.txt')
        logger.info(f"Detection: {self.cfg.MODEL.DETECTION}")

    def load_dataset(self):
        dataset_mode = ['train', 'eval'] if self.cfg.MODEL.MODE == 'train' else ['test']
        dataset_name = self.cfg.DATASET.NAME

        dataset = [BaseGraphDataset(self.cfg, self.cfg.DATASET.SEQUENCE, mode, self.feature_extractor, osp.join(self.cfg.DATASET.DIR, dataset_name))
                       for mode in dataset_mode]
        logger.info(f"Loading {dataset_name} sequences {dataset[0].seq_names}")
        logger.info(f"Total graphs for training: {len(dataset[0])} and validating: {len(dataset[1])}\n"
                    if self.cfg.MODEL.MODE == 'train' else f"Total graphs for testing: {len(dataset[0])}\n")
        return dataset

    def load_param(self, mode):
        if mode == 'test':
            ckpt = torch.load(self.cfg.TEST.CKPT_FILE_SG)
            self.SG = {'node_feature_encoder': NodeFeatureEncoder(self.cfg, in_dim=512),
                       'edge_feature_encoder': EdgeFeatureEncoder(self.cfg, in_dim=4),
                       'mpn': MPN(self.cfg),
                       'predictor': EdgePredictor(self.cfg)}
            self.SG['node_feature_encoder'].load_state_dict(ckpt['node_feature_encoder'])
            self.SG['edge_feature_encoder'].load_state_dict(ckpt['edge_feature_encoder'])
            self.SG['mpn'].load_state_dict(ckpt['mpn'])
            self.SG['predictor'].load_state_dict(ckpt['predictor'])
            logger.info(f'Load Spatial Graph param from {self.cfg.TEST.CKPT_FILE_SG}')

            ckpt = torch.load(self.cfg.TEST.CKPT_FILE_TG)
            self.TG = {'node_feature_encoder': NodeFeatureEncoder(self.cfg, in_dim=516),
                       'edge_feature_encoder': EdgeFeatureEncoder(self.cfg, in_dim=6),
                       'mpn': MPN(self.cfg),
                       'predictor': EdgePredictor(self.cfg)}
            self.TG['node_feature_encoder'].load_state_dict(ckpt['node_feature_encoder'])
            self.TG['edge_feature_encoder'].load_state_dict(ckpt['edge_feature_encoder'])
            self.TG['mpn'].load_state_dict(ckpt['mpn'])
            self.TG['predictor'].load_state_dict(ckpt['predictor'])
            logger.info(f'Load Temporal Graph param from {self.cfg.TEST.CKPT_FILE_TG}')

        else:
            ckpt = torch.load(self.cfg.MODEL.LAST_CKPT_FILE)
            logger.info(f'Continue training. Load param from {self.cfg.MODEL.LAST_CKPT_FILE}')

            self.node_feature_encoder.load_state_dict(ckpt['node_feature_encoder'])
            self.edge_feature_encoder.load_state_dict(ckpt['edge_feature_encoder'])
            self.mpn.load_state_dict(ckpt['mpn'])
            self.predictor.load_state_dict(ckpt['predictor'])

        return ckpt

    def train(self):
        optim = torch.optim.Adam(
            [{"params": self.node_feature_encoder.parameters()},
             {"params": self.edge_feature_encoder.parameters()},
             {"params": self.mpn.parameters()},
             {"params": self.predictor.parameters()}],
            lr=self.cfg.SOLVER.LR
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.cfg.SOLVER.EPOCHS, eta_min=0.001)
        scheduler_warmup = GradualWarmupScheduler(optim, 1.0, 10, scheduler_cosine)
        optim.zero_grad()
        optim.step()

        train_dataset, eval_dataset = self.load_dataset()
        train_loader = DataLoader(train_dataset, self.cfg.SOLVER.BATCH_SIZE, shuffle=True, collate_fn=udf_collate_fn, drop_last=True)
        eval_loader = DataLoader(eval_dataset, self.cfg.SOLVER.BATCH_SIZE, collate_fn=udf_collate_fn)
        writer = SummaryWriter(self.output_dir)

        if self.cfg.MODEL.RESUME:
            _ = self.load_param(self.cfg.MODEL.MODE)

        logger.info("Training begin...")
        for epoch in range(self.cfg.SOLVER.EPOCHS):
            self._train_one_epoch(epoch, train_loader, optim, scheduler_warmup, writer)
            if epoch % self.cfg.SOLVER.EVAL_EPOCH == 0 :
                self._eval_one_epoch(epoch, eval_loader, writer, self.output_dir) # save in _eval_
        writer.close()

    def test(self):
        ckpt = self.load_param(self.cfg.MODEL.MODE)

        visualize_dir = None
        if self.cfg.OUTPUT.VISUALIZE:
            visualize_dir = osp.join(self.output_dir, 'visualize')
            make_dir(visualize_dir)

        test_dataset = self.load_dataset()[0]
        test_loader = DataLoader(test_dataset, 1, collate_fn=udf_collate_fn)
        self._test_one_epoch(test_loader, ckpt['L'], visualize_dir) # generate inference file

        # evaluate using inference file
        logger.info('Evaluation Result:')
        evaluate(self.cfg, self.output_dir)

    def _train_one_epoch(self, epoch: int, dataloader, optimizer, scheduler, writer):
        scheduler.step()
        losses = []
        for i, data in enumerate(dataloader):
            graph_losses = []
            if self.cfg.OUTPUT.LOG:
                logger.info(f'{len(data)} datas loaded.')
            for graph, node_feature, edge_feature, y_true in data:
                x_node = self.node_feature_encoder(node_feature)
                x_edge = self.edge_feature_encoder(edge_feature)
                step_losses = []
                for _ in range(self.cfg.SOLVER.MAX_PASSING_STEPS):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)
                    y_pred = self.predictor(x_edge)
                    step_loss = sigmoid_focal_loss(y_pred, y_true,
                                                   self.cfg.SOLVER.FOCAL_ALPHA, self.cfg.SOLVER.FOCAL_GAMMA, "mean")
                    step_losses.append(step_loss)
                graph_loss = sum(step_losses)
                graph_losses.append(graph_loss)

            loss = sum(graph_losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item() / self.cfg.SOLVER.BATCH_SIZE
            losses.append(loss_val)
            if self.cfg.OUTPUT.LOG:
                logger.info(f"epoch=({epoch}/{self.cfg.SOLVER.EPOCHS - 1})"
                            f" | [{i + 1}/{len(dataloader)}]"
                            f" | graph_loss={loss_val:.4f}")

        avg_loss = sum(losses) / len(losses)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        if self.cfg.OUTPUT.LOG:
            logger.opt(ansi=True).info(f"<fg 255,204,0>【Finished training epoch {epoch}】 avg_train_loss={avg_loss:.4f}</fg 255,204,0>")

    def _eval_one_epoch(self, epoch: int, dataloader, writer, output_dir):
        losses = []
        for i, data in enumerate(dataloader):
            graph_losses = []
            for graph, node_feature, edge_feature, y_true in data:
                x_node = self.node_feature_encoder(node_feature)
                x_edge = self.edge_feature_encoder(edge_feature)
                step_losses = []
                for _ in range(self.cfg.SOLVER.MAX_PASSING_STEPS):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)
                    y_pred = self.predictor(x_edge)
                    step_loss = sigmoid_focal_loss(y_pred, y_true,
                                                   self.cfg.SOLVER.FOCAL_ALPHA, self.cfg.SOLVER.FOCAL_GAMMA, "mean")
                    step_losses.append(step_loss)
                graph_loss = sum(step_losses)
                graph_losses.append(graph_loss)

            loss = sum(graph_losses)
            loss_val = loss.item() / self.cfg.SOLVER.BATCH_SIZE
            losses.append(loss_val)
            if self.cfg.OUTPUT.LOG:
                logger.info(f"epoch=({epoch}/{self.cfg.SOLVER.EPOCHS - 1})"
                            f" | [{i + 1}/{len(dataloader)}]"
                            f" | graph_loss={loss_val:.4f}")

        avg_loss = sum(losses) / len(losses)
        writer.add_scalar("Loss/validation", avg_loss, epoch)
        if self.cfg.OUTPUT.LOG:
            logger.opt(ansi=True).info(f"<fg 255,153,51>【Finished eval epoch {epoch}】 avg_eval_loss={avg_loss:.6f}</fg 255,153,51>")

        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self._save_one_epoch(epoch, output_dir)

    @torch.no_grad()
    def _test_one_epoch(self, dataloader, max_passing_steps: int, visualize_output_dir=None):
        pre_TG = None
        for i, data in enumerate(dataloader):
            """ Spatial Graph
                Input: detection from each camera at current frame
                Output: A aggregated graph with nodes only
            """
            for graph, node_feature, edge_feature in data:
                x_node = self.SG['node_feature_encoder'](node_feature)
                x_edge = self.SG['edge_feature_encoder'](edge_feature)
                for _ in range(max_passing_steps):
                    x_node, x_edge = self.SG['mpn'](graph, x_node, x_edge)
                y_pred = self.SG['predictor'](x_edge)
                SG = self.tracklet.inference(i, graph, y_pred, 'SG') # post-processing & graph reconfiguration
                if pre_TG is None:
                    pre_TG = SG # t=0

            if self.cfg.OUTPUT.LOG:
                logger.opt(ansi=True).info(f'<fg 255,204,0>Iteration {i}: Spatial Graph done.'+
                                        f'SG: {SG.num_nodes()} nodes and {SG.num_edges()} edges.</fg 255,204,0>')


            """ Temporal Graph
                Input: SG at time i, TG at time i-1
                Output: inference result at time i
            """
            if i > 0:
                # Add edge (both input graphs are node-only)
                TG, node_feature, edge_feature = self.reconfiguration(pre_TG, SG)

                # Run Temporal Graph
                x_node = self.TG['node_feature_encoder'](node_feature)
                x_edge = self.TG['edge_feature_encoder'](edge_feature)
                for _ in range(max_passing_steps):
                    x_node, x_edge = self.TG['mpn'](TG, x_node, x_edge) # node: 32D, edge: 6D

                y_pred = self.TG['predictor'](x_edge)

                # Post-processing
                TG = self.tracklet.inference(i, TG, y_pred, 'TG') # post-processing
                pre_TG = TG

                if self.cfg.OUTPUT.VISUALIZE:
                    self.tracklet.visualize(i, visualize_output_dir)

                if self.cfg.OUTPUT.LOG:
                    logger.opt(ansi=True).info(f'<fg 255,204,0>Iteration {i}: Temporal Graph done.'+
                                            f'TG: {TG.num_nodes()} nodes and {TG.num_edges()} edges.</fg 255,204,0>')
            if self.cfg.OUTPUT.LOG:
                logger.opt(ansi=True).info(f'<fg 255,153,51>【Finished inference iteration {i}/{len(dataloader)-1}】</fg 255,153,51>\n')

    def reconfiguration(self, pre_TG, SG):
        TG = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        n_node_preTG = pre_TG.num_nodes()
        n_node_SG = SG.num_nodes()
        TG.add_nodes(n_node_SG, SG.ndata)
        TG.add_nodes(n_node_preTG, pre_TG.ndata)

        g_fID = TG.ndata['fID']
        _now = int(max(g_fID))

        # remove old node
        li = torch.where(g_fID < _now - 2)[0]
        TG.remove_nodes(list(li))

        # add edge (revisied ver., don't connect past two nodes)
        _from, _to = [], []
        for n1 in range(TG.num_nodes()):
            if g_fID[n1] != _now:
                continue
            for n2 in range(TG.num_nodes()):
                if g_fID[n2] == _now:
                    continue
                _from.append(n1)
                _to.append(n2)
        TG.add_edges(_from + _to, _to + _from)

        g_fID = TG.ndata['fID']
        reid_feature = TG.ndata['feat']
        projs = TG.ndata['proj']
        velocitys = TG.ndata['velocity']
        g_cID = torch.ones(g_fID.shape).cuda()
        node_feature = torch.cat((reid_feature, projs, g_cID), 1) # 516 d

        u = TG.edges()[0].type(torch.long)
        v = TG.edges()[1].type(torch.long)

        edge_feature = torch.vstack((
            torch.pairwise_distance(reid_feature[u], reid_feature[v]).to(self.device),
            1 - torch.cosine_similarity(reid_feature[u], reid_feature[v]).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device),
            torch.pairwise_distance(velocitys[u, :2], velocitys[v, :2], p=1).to(self.device),
            torch.pairwise_distance(velocitys[u, :2], velocitys[v, :2], p=2).to(self.device),
        )).T
        TG.edata['embed'] = edge_feature

        return TG, node_feature, edge_feature

    def _save_one_epoch(self, epoch: int, output_dir):
        cfg = self.cfg
        model_path = osp.join(output_dir,
                              f"{cfg.DATASET.NAME}_{cfg.DATASET.SEQUENCE[0]}_{cfg.SOLVER.TYPE}_epoch{epoch}.pth")
        torch.save({
            "node_feature_encoder": self.node_feature_encoder.state_dict(),
            "edge_feature_encoder": self.edge_feature_encoder.state_dict(),
            "mpn": self.mpn.state_dict(),
            "predictor": self.predictor.state_dict(),
            "L": self.cfg.SOLVER.MAX_PASSING_STEPS
        }, model_path)
        if self.cfg.OUTPUT.LOG:
            logger.info(f"Model has been saved in {model_path}.\n")