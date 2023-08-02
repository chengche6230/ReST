import os.path as osp
import json
import cv2
import copy
from loguru import logger
from tqdm import trange

import dgl
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F


class BaseGraphDataset(Dataset):
    """Class for Graph Dataset."""

    def __init__(self, cfg, seq_names: list, mode: str, feature_extractor, dataset_dir: str):
        assert mode in ("train", "eval", "test")
        assert len(seq_names) != 0

        self.cfg = cfg
        self.seq_names = seq_names
        self.mode = mode
        self.device = self.cfg.MODEL.DEVICE
        self.feature_extractor = feature_extractor
        self.dataset_dir = dataset_dir

        self._H = []  # homography matrices, H[seq_id][cam_id] => torch.Tensor(3*3)
        self._P = []  # images name pattern, F[seq_id][cam_id] => image path pattern
        self._S = []  # frames annotation in sequences, S[seq_id] => frame based dict (key type: str)
        self._SFI = None  # a (N*2) size tensor, store <seq_id, frame_id>
        self.load_dataset()

        self.frame_range = [self._SFI[0][1], self._SFI[-1][1]]

        self.graph = None # for inference
        self.chunks = [] # for training, use fid as index

        self.nodes = [] # use fid as index
        self.load_chunks()

    def load_dataset(self):
        with open(osp.join(self.dataset_dir, 'metainfo.json')) as fp:
            meta_info = json.load(fp)

        if len(self.seq_names) == 1 and self.seq_names[0] == 'all':
            self.seq_names = list(meta_info.keys())

        SFI = []
        for seq_id, name in enumerate(self.seq_names):
            output_path = osp.join(self.dataset_dir, name, 'output', f'{self.cfg.MODEL.DETECTION}_{self.mode}.json')
            with open(output_path, 'r') as fp:
                frames = json.load(fp)
            frames_id = list(map(int, frames.keys()))
            f_idx = torch.tensor(frames_id, dtype=torch.int32).unsqueeze(1)
            s_idx = torch.full_like(f_idx, seq_id)
            SFI.append(torch.hstack([s_idx, f_idx]))
            self._S.append(frames)
            self._H.append(torch.tensor(meta_info[name]['homography']))
            self._P.append([f'{self.dataset_dir}/{name}/output/frames/{{}}_{i}.jpg'
                            for i in range(meta_info[name]['cam_nbr'])])
        self._SFI = torch.vstack(SFI)

    def load_chunks(self):
        num_frames = self._SFI.shape[0]
        sid = 0

        for t in trange(num_frames):
        # for t in trange(110):
            sid, fid = tuple(map(int, self._SFI[t]))
            frame_images = self.load_images(sid, fid) # {cameras} images with [C, H, W]
            frames = torch.tensor(self._S[sid][str(fid)], dtype=torch.int32)
            n_node = frames.shape[0] # number of detection

            ## Nodes
            # attributes in node
            projs = torch.zeros(n_node, 3, dtype=torch.float32)
            (H, W) = self.cfg.FE.INPUT_SIZE
            bdets = torch.zeros(n_node, 3, H, W, dtype=torch.float32) # (N, C, H, W)
            bboxs = torch.zeros(n_node, 4, dtype=torch.float32) # x, y, w, h
            cIDs = torch.zeros(n_node, 1, dtype=torch.int8) # camera ID
            fIDs = torch.zeros(n_node, 1, dtype=torch.int16) # frame ID (timestamp)
            tIDs = torch.zeros(n_node, 1, dtype=torch.int16) # track ID (person ID)
            tIDs_pred = torch.zeros(n_node, 1, dtype=torch.int16) # track ID (person ID) for inference

            for n in range(n_node):
                tid, cid = frames[n, -2:]
                x, y, w, h = frames[n, :4]

                # projection for Wildtrack
                if self.cfg.DATASET.NAME in ['Wildtrack']:
                    proj = torch.matmul(torch.linalg.inv(self._H[sid][cid]),
                                        torch.t(torch.tensor([x + w / 2, y + h, 1], dtype=torch.float32)))
                # projection for other datasets
                else:
                    proj = torch.matmul(self._H[sid][cid],
                                        torch.tensor([x + w / 2, y + h, 1], dtype=torch.float32))
                projs[n] = proj / proj[-1]

                det = frame_images[int(cid)][:, y: y + h, x: x + w]
                det = T.Resize((H, W))(det)
                bdets[n] = det
                bboxs[n] = frames[n, :4]
                cIDs[n] = cid
                fIDs[n] = fid
                tIDs[n] = tid
                tIDs_pred[n] = -1 # default

            if self.cfg.FE.CHOICE == 'CNN':
                # Original CNN re-ID feature extractor
                det_feature = self.feature_extractor(bdets)  # (N, 512)

            nodes_attr = {'cID': cIDs.to(self.device), # [1]
                          'fID': fIDs.to(self.device), # [1]
                          'tID': tIDs.to(self.device), # [1]
                          'tID_pred': tIDs_pred.to(self.device), # [1]
                          'bbox': bboxs.to(self.device), # [4], bbox info. (x, y, w, h)
                          'feat': det_feature.to(self.device), # [512], re-ID(appearance) feature
                          'proj': projs.to(self.device) # [3], geometric position
                         }
            self.nodes.append(nodes_attr)

        ## Chunks, one chunk represent one training data
        if self.mode != 'test':
            if self.cfg.SOLVER.TYPE == 'SG':
                for i in range(num_frames-1):
                    graph = dgl.graph(([], []), idtype=torch.int32, device=self.device)
                    for j in range(2):
                        n_node = len(self.nodes[i+j]['cID'])
                        nodes_attr = self.nodes[i+j]
                        graph.add_nodes(n_node, nodes_attr)
                    if graph.num_nodes() != 0:
                        c = int(graph.ndata['cID'][0])
                        li = torch.where(graph.ndata['cID']==c)[0]
                        if len(li) != graph.num_nodes():
                            self.chunks.append(graph)
            else:
                w_size = 2 # temporal window size
                for i in range(num_frames - w_size + 1):
                    for c in range(self.cfg.DATASET.CAMS):
                        graph = dgl.graph(([], []), idtype=torch.int32, device=self.device)
                        for j in range(w_size):
                            li = torch.where(self.nodes[i+j]['cID']==c)[0]
                            n_node = len(li)

                            ## Velocity
                            velocity = torch.zeros(n_node, 2, dtype=torch.float32) # Vx, Vy
                            if True:
                                pre_li = torch.where(self.nodes[i+j-1]['cID']==c)[0]
                                pre_tID = self.nodes[i+j-1]['tID'][pre_li]
                                pre_proj = self.nodes[i+j-1]['proj'][pre_li]
                                tID = self.nodes[i+j]['tID'][li]
                                proj = self.nodes[i+j]['proj'][li]
                                for n in range(n_node):
                                    pre_idx = torch.where(pre_tID==tID[n])[0]
                                    if len(pre_idx) != 0:
                                        velocity[n] = proj[n][:2] - pre_proj[pre_idx][0][:2]

                            nodes_attr = {'cID': self.nodes[i+j]['cID'][li],
                                          'fID': self.nodes[i+j]['fID'][li],
                                          'tID': self.nodes[i+j]['tID'][li],
                                          'tID_pred': self.nodes[i+j]['tID_pred'][li],
                                          'bbox': self.nodes[i+j]['bbox'][li],
                                          'feat': self.nodes[i+j]['feat'][li],
                                          'proj': self.nodes[i+j]['proj'][li],
                                          'velocity': velocity
                                          }
                            graph.add_nodes(n_node, nodes_attr)
                        if graph.num_nodes() != 0:
                            f = int(graph.ndata['fID'][0])
                            li = torch.where(graph.ndata['fID']==f)[0]
                            if len(li) != graph.num_nodes():
                                self.chunks.append(graph)

    def load_images(self, seq_id: int, frame_id: int, tensor=True):
        imgs = []
        for img_path in self._P[seq_id]:
            img = cv2.imread(img_path.format(frame_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if tensor:
                img = T.ToTensor()(img)  # (C, H, W), float
            else:
                img = torch.from_numpy(img)
                img = torch.permute(img, (2, 0, 1))  # (C, H, W), uint8
            imgs.append(img)
        return imgs

    def __len__(self):
        if self.mode != 'test':
            return len(self.chunks)
        else:
            return self._SFI.shape[0] # length of frames

    def __getitem__(self, index):
        if self.mode == 'test':
            return self.__getInference__(index)

        g = copy.deepcopy(self.chunks[index])

        # Data Augmentation of SG: randomly rm few cams nodes
        aug_dataset = ['Wildtrack', 'CAMPUS', 'PETS09', 'CityFlow']
        DN = self.cfg.DATASET.NAME
        if self.cfg.SOLVER.TYPE == 'SG' and DN in aug_dataset:
            if DN == 'Wildtrack':
                num_rm = 2
            else:
                num_rm = 1
            c = torch.randint(self.cfg.DATASET.CAMS, (num_rm,))
            for i in range(num_rm):
                li = torch.where(g.ndata['cID']==int(c[i]))[0]
                g.remove_nodes(list(li))

        # add edge
        u, v = [], []
        for n in range(g.num_nodes()):
            u += [n] * g.num_nodes()
            v += list(range(g.num_nodes()))
        g.add_edges(u, v)

        g_fID = g.ndata['fID']
        g_cID = g.ndata['cID']
        g_tID = g.ndata['tID']
        _from = g.edges()[0].type(torch.long)
        _to = g.edges()[1].type(torch.long)
        li = []
        if self.cfg.SOLVER.TYPE == 'SG':
            li = torch.where(g_cID[_from]==g_cID[_to])[0]
        else:
            li = torch.where(g_fID[_from]==g_fID[_to])[0]
        assert len(li) != 0
        g.remove_edges(list(li))

        reid_feature = g.ndata['feat']
        projs = g.ndata['proj']
        if self.cfg.SOLVER.TYPE == 'SG':
            node_feature = reid_feature
        else:
            node_feature = torch.cat((reid_feature, projs, g_cID), 1)

        u = g.edges()[0].type(torch.long)
        v = g.edges()[1].type(torch.long)
        if self.cfg.SOLVER.TYPE == 'SG':
            edge_feature = torch.vstack((
                torch.pairwise_distance(reid_feature[u], reid_feature[v]).to(self.device),
                1 - torch.cosine_similarity(reid_feature[u], reid_feature[v]).to(self.device),
                torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),
                torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device)
            )).T  # (E, 4)
        else: #TG
            velocitys = g.ndata['velocity']
            edge_feature = torch.vstack((
                torch.pairwise_distance(reid_feature[u], reid_feature[v]).to(self.device),
                1 - torch.cosine_similarity(reid_feature[u], reid_feature[v]).to(self.device),
                torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),
                torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device),
                torch.pairwise_distance(velocitys[u, :2], velocitys[v, :2], p=1).to(self.device),
                torch.pairwise_distance(velocitys[u, :2], velocitys[v, :2], p=2).to(self.device),
            )).T # (E, 6)
        g.edata['embed'] = edge_feature

        # Groundtruth y label (E, 1)
        g_tID = g.ndata['tID']
        _from = g.edges()[0].type(torch.long)
        _to = g.edges()[1].type(torch.long)
        y_true = (g_tID[_from] == g_tID[_to]).type(torch.float16)
        assert g.num_edges() != 0
        assert g.num_edges() == y_true.shape[0]

        return g, node_feature, edge_feature, y_true

    def __getInference__(self, index):
        # for spatial graph only in inference stage.
        sid, fid = tuple(map(int, self._SFI[index]))
        frames = torch.tensor(self._S[sid][str(fid)], dtype=torch.int32)
        n_node = frames.shape[0]

        self.graph = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        g = self.graph

        # add new detections(nodes)
        g.add_nodes(n_node, self.nodes[index])

        # add edge
        u, v = [], []
        for n in range(g.num_nodes()):
            u += [n] * g.num_nodes()
            v += list(range(g.num_nodes()))
        g.add_edges(u, v)

        g_cID = g.ndata['cID']
        _from = g.edges()[0].type(torch.long)
        _to = g.edges()[1].type(torch.long)
        li = torch.where(g_cID[_from]==g_cID[_to])[0]
        assert len(li) != 0
        g.remove_edges(list(li))

        node_feature = g.ndata['feat']
        projs = g.ndata['proj']
        u = g.edges()[0].type(torch.long)
        v = g.edges()[1].type(torch.long)

        edge_feature = torch.vstack((
            torch.pairwise_distance(node_feature[u], node_feature[v]).to(self.device),
            1 - torch.cosine_similarity(node_feature[u], node_feature[v]).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device)
        )).T  # (E, 4)
        g.edata['embed'] = edge_feature

        if self.cfg.OUTPUT.LOG:
            logger.info(f'Loaded graph with {g.num_nodes()} nodes and {g.num_edges()} edges.')
        return self.graph, node_feature, edge_feature
