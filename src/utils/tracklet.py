import os, copy
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from itertools import count  # for SG visualization, do not remove
from loguru import logger

import dgl
import networkx as nx

import torch

# for visualization
COLORS = [[39, 188, 221], [167, 72, 214], [82, 67, 198], [76, 198, 232],
          [137, 24, 13], [142, 31, 221], [47, 196, 154], [40, 110, 201],
          [10, 147, 115], [71, 4, 216], [85, 113, 224], [41, 173, 118],
          [52, 172, 237], [80, 237, 164], [175, 164, 65], [70, 53, 178],
          [39, 135, 4], [242, 55, 201], [221, 31, 180], [89, 224, 170],
          [117, 21, 43], [34, 205, 214], [114, 244, 22], [181, 126, 39],
          [127, 17, 69], [102, 12, 211], [26, 178, 127], [198, 67, 249],
          [96, 45, 6], [165, 104, 58]]


def save_graph(cfg, graph, file_name, now=None):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph, weight='y_pred')

    ## Spatial Grpha (colored by tID)
    # groups = set(nx.get_node_attributes(graph,'tID').values())
    # mapping = dict(zip(sorted(groups),count()))
    # nodes = graph.nodes()
    # colors = [mapping[graph.nodes[n]['tID']] for n in nodes]
    # nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, cmap=plt.cm.jet)

    ## Temporal Graph (colored by fID)
    nodes = graph.nodes()
    colors = []
    for n in nodes:
        if graph.nodes[n]['fID'] == now:
            colors.append(1)
        else:
            colors.append(0)
    nx.draw_networkx_nodes(graph,
                           pos,
                           nodelist=nodes,
                           node_color=colors,
                           cmap=plt.cm.jet)


    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig


class Tracklet():
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.outpu_dir = output_dir
        self.graph = None
        self.y_pred = None
        self.newID = 0

    def inference(self, time, graph, y_pred, mode):
        self.time = time
        self.graph = graph
        self.y_pred = y_pred
        self.graph.edata['y_pred'] = self.y_pred

        graph = None
        if mode == 'SG':
            self.remove_edge()
            self.postprocessing_sg()
            graph = self.aggregating_sg()

        if mode == 'TG':
            self.remove_edge()
            self.postprocessing_tg()
            self.assign_ID()
            self.write_infer_file()
            graph = self.aggregating_tg()

        return graph


    def remove_edge(self):
        if self.graph is None or self.y_pred is None:
            logger.error(f'graph or y_pred is none.')

        edge_thresh = self.cfg.TEST.EDGE_THRESH
        # if self.cfg.OUTPUT.LOG:
        #     print(max(self.y_pred).item(), min(self.y_pred).item())
        g = self.graph
        orig_edge = g.num_edges()
        li = torch.where(self.y_pred < edge_thresh)[0]
        if len(li) > 0:
            g.remove_edges(list(li))

        # remove multi-edge(more than one edge b/t two nodes)
        edge_set = set()
        rm_li = []
        _from = g.edges()[0].type(torch.long)
        _to = g.edges()[1].type(torch.long)
        for i in range(g.num_edges()):
            f = _from[i].item()
            t = _to[i].item()
            if (f, t) in edge_set:
                rm_li.append(i)
            else:
                edge_set.add((f, t))
                edge_set.add((t, f))
        # logger.info(f'Removed {len(rm_li)} multi-edges. Graph: {g.num_nodes()} nodes, {g.num_edges()} edges.')
        g.remove_edges(rm_li)

    def postprocessing_sg(self):
        # 1. Eq.(11): degree(v) < C-1   -> splitting_1()
        # 2. Eq.(10): |V(H)| < C        -> splitting_2()
        rm_len = 0

        rm_len += self.splitting_1()  ##degree

        rml, again = self.splitting_2()  ## number of connected components
        rm_len += rml

        if self.cfg.OUTPUT.LOG:
            logger.info(
                f'Post-processing done, removed {rm_len} edges. Graph: {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges.'
            )

    def toSimpleGraph(self, mg):
        G = nx.Graph()
        for u, v, data in mg.edges(data=True):
            w = data['y_pred'] if 'y_pred' in data else 1.0
            G.add_edge(u, v, weight=w)
        return G

    def findEdgebyNode(self, u, v):
        _from = self.graph.edges()[0].type(torch.long)
        _to = self.graph.edges()[1].type(torch.long)
        fli = torch.where(_from == u)[0].tolist()
        tli = torch.where(_to == v)[0].tolist()
        eid = list(set(fli).intersection(tli))

        if len(eid) == 0:
            fli = torch.where(_from == v)[0].tolist()
            tli = torch.where(_to == u)[0].tolist()
            eid = list(set(fli).intersection(tli))

        return eid[0]

    def findMinEdge(self, edges, g_ypred):
        min_edge = -1
        min_pred = 1.1
        rm_f, rm_t = -1, -1
        for f, t in edges:
            eid = self.findEdgebyNode(f, t)
            if g_ypred[eid] < min_pred:
                min_pred = g_ypred[eid]
                min_edge = eid
                rm_f = f
                rm_t = t
        return min_edge, rm_f, rm_t

    def splitting_1(self):
        nxg = dgl.to_networkx(self.graph.cpu(),
                              edge_attrs=['y_pred']).to_undirected()
        nxg = self.toSimpleGraph(nxg)

        g_ypred = self.graph.edata['y_pred']
        rm_li = []
        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            sg = nxg.subgraph(cc)
            flows = [d for n, d in sg.degree(ccli)]
            violate = torch.where(
                torch.tensor(flows) > self.cfg.DATASET.CAMS - 1)[0]
            while len(violate) > 0:
                bridge = list(nx.bridges(sg))
                if len(bridge) == 1:
                    rm_f, rm_t = bridge[0]
                    rm_li.append(self.findEdgebyNode(rm_f, rm_t))
                elif len(bridge) > 1:  # more than one bridge
                    rm_eid, rm_f, rm_t = self.findMinEdge(bridge, g_ypred)
                    rm_li.append(rm_eid)
                else:  # no bridge
                    rm_eid, rm_f, rm_t = self.findMinEdge(sg.edges(), g_ypred)
                    rm_li.append(rm_eid)
                sg = nx.Graph(sg)
                sg.remove_edge(rm_f, rm_t)
                flows = [d for n, d in sg.degree(ccli)]
                violate = torch.where(
                    torch.tensor(flows) > self.cfg.DATASET.CAMS - 1)[0]

        self.graph.remove_edges(rm_li)
        # logger.info(f'Pruned {len(rm_li)} edges. Graph: {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges.')
        return len(rm_li)

    def splitting_2(self):
        nxg = dgl.to_networkx(self.graph.cpu(),
                              edge_attrs=['y_pred']).to_undirected()
        nxg = self.toSimpleGraph(nxg)

        g_ypred = self.graph.edata['y_pred']
        rm_li = []
        again = False
        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            sg = nxg.subgraph(cc)
            violate = True if len(ccli) > self.cfg.DATASET.CAMS else False
            while violate:  # violate condition
                bridge = list(nx.bridges(sg))
                if len(bridge) == 1:
                    rm_f, rm_t = bridge[0]
                    rm_li.append(self.findEdgebyNode(rm_f, rm_t))
                elif len(bridge) > 1:
                    rm_eid, rm_f, rm_t = self.findMinEdge(bridge, g_ypred)
                    rm_li.append(rm_eid)
                else:  # no bridge
                    rm_eid, rm_f, rm_t = self.findMinEdge(sg.edges(), g_ypred)
                    rm_li.append(rm_eid)

                sg = nx.Graph(sg)
                sg.remove_edge(rm_f, rm_t)
                cnt = 0
                error = False
                for i, cc in enumerate(nx.connected_components(sg)):
                    cnt += 1
                    if len(list(cc)) > self.cfg.DATASET.CAMS:
                        error = True
                if cnt > 1:  # successfully break into two c.c
                    if error:  # but still violate, do whole function again
                        again = True
                    break
        self.graph.remove_edges(rm_li)
        # logger.info(f'Split {len(rm_li)} edges. Graph: {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges.')
        return len(rm_li), again

    def aggregating_sg(self):
        nxg = dgl.to_networkx(self.graph.cpu(),
                              node_attrs=['tID']).to_undirected()

        # visualize graph
        # save_graph(self.cfg, nxg, os.path.join(self.outpu_dir,'visualize', f'cluster_{self.time}.png'), now=self.time+self.cfg.TEST.FRAME_START)
        # nxg = self.toSimpleGraph(nxg) # make cc w/ only one node disappear, decrease Wildtrack performance a little

        g_tID = self.graph.ndata['tID']
        g_cID = self.graph.ndata['cID']
        g_feat = self.graph.ndata['feat']
        g_proj = self.graph.ndata['proj']
        g_bbox = self.graph.ndata['bbox']

        n_node = 0
        for i, cc in enumerate(nx.connected_components(nxg)):
            n_node += 1
        fIDs = torch.zeros(n_node, 1, dtype=torch.int16)
        feats = torch.zeros(n_node, 512, dtype=torch.float32)
        projs = torch.zeros(n_node, 3, dtype=torch.float32)
        velocitys = torch.zeros(n_node, 2, dtype=torch.float32)
        tIDs_pred = torch.zeros(n_node, 1, dtype=torch.int16)
        self.bboxs = []

        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            fIDs[i] = self.time + self.cfg.TEST.FRAME_START
            feats[i] = torch.mean(g_feat[ccli], 0)
            projs[i] = torch.mean(g_proj[ccli], 0)
            tIDs_pred[i] = -1

            # save bbox and cID for inference
            tmp_li = []
            for node in ccli:
                tmp_li.append([
                    g_cID[node].item(),
                    [g_bbox[node][i].item() for i in range(4)]
                ])
            self.bboxs.append(tmp_li)

        nodes_attr = {
            'fID': fIDs.to(self.device),  # current time
            'feat': feats.to(self.device),  # mean feature of all nodes
            'proj': projs.to(self.device),  # mean projection of all nodes
            'velocity':
            velocitys.to(self.device),  # init v of spatial node to 0
            'tID_pred': tIDs_pred.to(self.device)  # initialize for inference
        }

        # create SG(node-only)
        sg = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        sg.add_nodes(n_node, nodes_attr)

        return sg

    def postprocessing_tg(self):
        # Eq.(12): degree(v) < M-1
        nxg = dgl.to_networkx(self.graph.cpu(),
                              edge_attrs=['y_pred']).to_undirected()
        nxg = self.toSimpleGraph(nxg)

        g_ypred = self.graph.edata['y_pred']
        rm_li = []

        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            sg = nxg.subgraph(cc)
            flows = [d for n, d in sg.degree(ccli)]
            violate = torch.where(torch.tensor(flows) > 1)[0]
            while len(violate) > 0:  # solve many-to-one violation
                rm_eid, rm_f, rm_t = self.findMinEdge(sg.edges(), g_ypred)
                rm_li.append(rm_eid)

                sg = nx.Graph(sg)
                sg.remove_edge(rm_f, rm_t)
                flows = [d for n, d in sg.degree(ccli)]
                violate = torch.where(torch.tensor(flows) > 1)[0]
        self.graph.remove_edges(rm_li)
        if self.cfg.OUTPUT.LOG:
            logger.info(
                f'Post-processing done, removed {len(rm_li)} edges. Graph: {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges.'
            )

    def assign_ID(self):
        g_tID_pred = self.graph.ndata['tID_pred']
        nxg = dgl.to_networkx(
            self.graph.cpu(), node_attrs=['fID'], edge_attrs=['y_pred']
        ).to_undirected()  # node_attrs=['tID_pred'], edge_attrs=['y_pred']
        if self.cfg.OUTPUT.VISUALIZE:
            save_graph(self.cfg,
                       nxg,
                       os.path.join(self.outpu_dir, 'visualize',
                                    f'cluster_{self.time}.png'),
                       now=self.time + self.cfg.TEST.FRAME_START)

        g_fID = self.graph.ndata['fID']
        for i, cc in enumerate(nx.connected_components(nxg)):
            nodes = list(cc)
            labeled_nodes = torch.where(
                g_tID_pred[nodes] != -1)[0]  # return list of index
            if len(labeled_nodes) > 0:
                label = copy.deepcopy(
                    g_tID_pred[nodes[labeled_nodes[0]]])
                self.graph.ndata['tID_pred'][nodes] = label
            else:
                label = self.newID
                self.graph.ndata['tID_pred'][nodes] = label
                self.newID += 1

    def write_infer_file(self):
        if self.time == 0:
            return
        attr = [
            'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
            'conf', 'x', 'y', 'z'
        ]
        if self.cfg.DATASET.NAME == 'Wildtrack':
            frame = f'0000{(self.time + self.cfg.TEST.FRAME_START)*5:04d}'
        else:
            frame = f'{self.time + self.cfg.TEST.FRAME_START:4d}'
        current_fID = self.time + self.cfg.TEST.FRAME_START

        g_fID = self.graph.ndata['fID']
        g_tID = self.graph.ndata['tID_pred']
        g_proj = self.graph.ndata['proj']

        dfs = []
        for _ in range(self.cfg.DATASET.CAMS):
            dfs.append(pd.DataFrame(columns=attr))

        for n in range(self.graph.num_nodes()):
            if g_fID[n] != current_fID:  # online method, only preocess current nodes
                continue
            for i in range(len(self.bboxs[n])):
                x, y, w, h = self.bboxs[n][i][1]
                c = self.bboxs[n][i][0]
                dfs[c] = dfs[c].append(
                    {
                        'frame': frame,
                        'id': g_tID[n].cpu().item(),
                        'bb_left': x,
                        'bb_top': y,
                        'bb_width': w,
                        'bb_height': h,
                        'conf': 1,
                        'x': -1,
                        'y': -1,
                        'z': -1
                    },
                    ignore_index=True)
        for c in range(self.cfg.DATASET.CAMS):
            dfs[c].to_csv(os.path.join(self.outpu_dir, f'c{c}.txt'),
                          header=None,
                          index=None,
                          sep=',',
                          mode='a')

        gp_dfs = pd.DataFrame(columns=attr)
        for n in range(self.graph.num_nodes()):
            if g_fID[n] != current_fID:  # online method, only preprocess current nodes
                continue
            y, x, _ = g_proj[n].cpu().tolist()
            gp_dfs = gp_dfs.append(
                {
                    'frame': frame,
                    'id': g_tID[n].cpu().item(),
                    'bb_left': -1,
                    'bb_top': -1,
                    'bb_width': -1,
                    'bb_height': -1,
                    'conf': 1,
                    'x': x,
                    'y': y,
                    'z': -1
                },
                ignore_index=True)
        gp_dfs.to_csv(os.path.join(self.outpu_dir, f'gp.txt'),
                      header=None,
                      index=None,
                      sep=',',
                      mode='a')
    def aggregating_tg(self):
        nxg = dgl.to_networkx(self.graph.cpu()).to_undirected()

        g_feat = self.graph.ndata['feat']
        g_proj = self.graph.ndata['proj']
        g_fID = self.graph.ndata['fID']
        g_tIDpred = self.graph.ndata['tID_pred']

        n_node = 0
        for i, cc in enumerate(nx.connected_components(nxg)):
            n_node += 1
        fIDs = torch.zeros(n_node, 1, dtype=torch.int16)
        feats = torch.zeros(n_node, 512, dtype=torch.float32)
        projs = torch.zeros(n_node, 3, dtype=torch.float32)
        tIDs_pred = torch.zeros(n_node, 1, dtype=torch.int16)
        velocitys = torch.zeros(n_node, 2, dtype=torch.float32)

        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            if len(ccli) == 1:  # no match now
                fIDs[i] = g_fID[ccli[0]]
            else:
                fIDs[
                    i] = self.time + self.cfg.TEST.FRAME_START  # successful match, update time
            feats[i] = torch.mean(g_feat[ccli], 0)
            projs[i] = torch.mean(g_proj[ccli], 0)
            tIDs_pred[i] = g_tIDpred[ccli][0].item()
            if len(ccli) == 2:
                _pre, _now = -1, -1
                if g_fID[ccli][0] < g_fID[ccli][1]:
                    _pre, _now = 0, 1
                elif g_fID[ccli][0] > g_fID[ccli][1]:
                    _pre, _now = 1, 0
                if _pre != -1 and _now != -1:
                    velocitys[
                        i] = g_proj[ccli][_now][:2] - g_proj[ccli][_pre][:2]

        nodes_attr = {
            'fID': fIDs.to(self.device),  # current time
            'feat': feats.to(self.device),  # mean feature of all nodes
            'proj': projs.to(self.device),  # mean projection of all nodes
            'velocity': velocitys.to(self.device),  # velocity
            'tID_pred': tIDs_pred.to(self.device)  # initialize for inference
        }

        # create pre-TG(node-only)
        pre_tg = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        pre_tg.add_nodes(n_node, nodes_attr)

        return pre_tg

    def visualize(self, time, vis_output_dir):
        frame = time + self.cfg.TEST.FRAME_START
        output_dir = os.path.join(vis_output_dir, 'frames')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            for c in range(self.cfg.DATASET.CAMS):
                os.mkdir(os.path.join(output_dir, f'c{c}'))

        g_fID = self.graph.ndata['fID']
        g_tID = self.graph.ndata['tID_pred']
        g_proj = self.graph.ndata['proj']

        c = self.cfg.DATASET

        cam_nodes = []
        for _ in range(c.CAMS):
            cam_nodes.append([])
        for n in range(self.graph.num_nodes()):
            if g_fID[n] != frame:
                continue
            tID = g_tID[n]
            proj = g_proj[n]
            for i in range(len(self.bboxs[n])):
                x, y, w, h = self.bboxs[n][i][1]
                ca = self.bboxs[n][i][0]
                cam_nodes[ca].append([x, y, w, h, tID, proj])

        # bird_view = np.zeros((1080, 1920, 3))

        for cam in range(c.CAMS):
            frame_img = os.path.join(c.DIR, c.NAME, c.SEQUENCE[0],
                                     'output/frames', f'{frame}_{cam}.jpg')
            img = cv2.imread(frame_img)
            for b in range(len(cam_nodes[cam])):
                bbox = cam_nodes[cam][b][:4]
                tID_pred = cam_nodes[cam][b][4]
                proj = cam_nodes[cam][b][5]

                # bbox and label with color
                color = (COLORS[tID_pred.item() % 30])
                cv2.rectangle(
                    img, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                    color, 5)  #2
                cv2.rectangle(img, (int(bbox[0]) - 5, int(bbox[1]) - 40),
                              (int(bbox[0]) + 60, int(bbox[1])), color, -1)  #2
                cv2.putText(img, f'{tID_pred.item()}',
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
                            cv2.LINE_AA)

                # bird view visualization
                # bird_view = cv2.circle(bird_view, ((int(proj[0])+1200)//2, int(proj[1])-140), 22, color, -1)

            cv2.imwrite(os.path.join(output_dir, f'c{cam}/{frame}.jpg'), img)

        # bird view visualization
        # path = os.path.join(output_dir, 'bird_view')
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # cv2.imwrite(os.path.join(path, f'{frame}.jpg'), bird_view)