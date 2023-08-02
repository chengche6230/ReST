# import dgl
# import dgl.function as df
import torch
import torch.nn as nn


class MPN(nn.Module):
    """Message Passing Neural Network."""

    def __init__(self, cfg, ckpt=None):
        super(MPN, self).__init__()
        # Learnable MLP message encoders
        self.node_msg_encoder = nn.Sequential(
            nn.Linear(38, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.edge_msg_encoder = nn.Sequential(
            nn.Linear(70, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
            nn.ReLU()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def message_udf(self, edges):
        """ User define function message function
        """
        edge_msg = self.edge_msg_encoder(
            torch.cat((edges.dst['x'], edges.src['x'], edges.data['x']), 1)
        )
        self.em = edge_msg

        # Aggregate adjacent node feature(32) & edge feature to itself (6) => AGgregateEdgeMessage
        node_msg = self.node_msg_encoder(
            torch.cat((edges.dst['x'], edge_msg), 1)
        )
        return {'agem': node_msg}

    def reduce_udf(self, nodes):
        """ Udf aggregating function
        """
        return {'nm': nodes.mailbox['agem'].sum(dim=1)}

    def forward(self, graph, x_node, x_edge):
        with graph.local_scope():
            graph.ndata['x'] = x_node
            graph.edata['x'] = x_edge

            graph.update_all(message_func=self.message_udf, reduce_func=self.reduce_udf)
            return graph.ndata['nm'], self.em

if __name__ == "__main__":
    test = MPN('cuda')