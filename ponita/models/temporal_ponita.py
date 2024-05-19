import torch
import torch.nn as nn
from ponita.models.ponita import PonitaFiberBundle
import torch.nn.functional as F
import math

class TemporalPonita(PonitaFiberBundle):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 args,
                 output_dim_vec=0,
                 radius=None,
                 num_ori=20,
                 basis_dim=None,
                 degree=3,
                 widening_factor=4,
                 layer_scale=None,
                 task_level='graph',
                 multiple_readouts=True,
                 **kwargs):
        
        super().__init__(input_dim,
                         hidden_dim,
                         output_dim,
                         num_layers,
                         output_dim_vec=output_dim_vec,
                         radius=radius,
                         num_ori=num_ori,
                         basis_dim=basis_dim,
                         degree=degree,
                         widening_factor=widening_factor,
                         layer_scale=layer_scale,
                         task_level=task_level,
                         multiple_readouts=multiple_readouts,
                         **kwargs)  
        
        self.num_layers = num_layers
        self.args = args
        self.num_land_marks = self.args.n_nodes    
        self.kernel_size = self.args.kernel_size
        self.stride =  self.args.stride
        self.padding = int((self.kernel_size - 1) / 2)
        
        self.tconv = TCNUnit(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding, dropout_rate = args.temporal_dropout_rate)

    def forward(self, graph):

        # Lift and compute invariants
        graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(graph.attr) * self.windowing_fn(graph.dists).unsqueeze(-2)
        fiber_kernel_basis = self.fiber_basis_fn(graph.fiber_attr)

        # Initial feature embeding
        x = self.x_embedder(graph.x)

        # Interaction + readout layers
        readouts = []

        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            
            # Perform spatial convolutions
            # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]
            x = interaction_layer(x, graph.edge_index, edge_attr=kernel_basis, fiber_attr=fiber_kernel_basis, batch=graph.batch)
            
            # Perform temporal convolution
            # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]    
            if readout_layer is None:
                x = self.conv1d(x, graph, False)
            
            # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]
            if readout_layer is not None: 
                x = self.conv1d(x, graph, True)
                # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]
                # Pool across spatial temporal graph
                x = self.TS_Pooling(x, graph)

                # x [batch_size, num_ori, hidden_dim]
                x = readout_layer(x)
                
                # x [n_frames_per_vid x n_nodes x batch_size, num_ori, n_classes]
                readouts.append(x)
        
        readout = sum(readouts) / len(readouts)

        
        # readout_scalar [n_frames_per_vid x n_nodes x batch_size, num_ori, n_classes]
        readout_scalar, readout_vec = torch.split(readout, [self.output_dim, self.output_dim_vec], dim=-1)

        # output scalar [batch_size, n_classes]
        output_scalar = self.scalar_readout_fn(readout_scalar, graph.batch)
        
        # This is none
        output_vector = self.vec_readout_fn(readout_vec, graph.ori_grid, graph.batch)

        # Return predictions
        return output_scalar, output_vector
      
    def TS_Pooling(self, x, graph):
        """ Perform temporal pooling on the time axis """
        x_agg = []
        start_idx = 0
        for n_frames in graph.n_frames:
            # Select range corresponding to graph (n_frames x n_nodes)
            _, num_ori, hidden_dim = x.shape
            n_idx = n_frames*self.num_land_marks
            x_tmp = x[start_idx:start_idx+n_idx,]
            '''
            method 1 just do it
            '''
            # Aggregate spatial dimension 
            x_tmp = x_tmp.mean(dim=0)
            '''
            method 2 split it TS
            '''
            #x_tmp = x_tmp.view(-1,self.num_land_marks, num_ori, hidden_dim)
            #x_tmp = x_tmp.mean(dim=0)
            #x_tmp = x_tmp.mean(dim=0) 
            
            x_agg.append(x_tmp)
            start_idx += n_frames*self.num_land_marks
        
        x_agg = torch.stack(x_agg, dim=0)  
        
        return x_agg
    
    
    
    def conv1d(self, x, graph, use_attn = False):
        """ Perform 1D convolution on the time axis 
        """

        x_conv = []
        start_idx = 0

        # For each graph in the batch
        for n_frames in graph.n_frames:
            
            # Select range corresponding to graph (n_frames x n_nodes)
            n_idx = n_frames*self.num_land_marks
            # TODO: Should this be n_frames or n_frames-1
            x_tmp = x[start_idx:start_idx+n_idx,]

            # Rearrange tensor
            # TODO: Check view vs permute vs restack
            num_nodes_batch, num_ori, num_channels = x_tmp.shape

            # [N_frames, num_landmarks*num_ori, num_channels]
            x_tmp = x_tmp.view(-1, self.num_land_marks*num_ori, num_channels)

            # [num_landmarks*num_ori, num_channels, N_frames]
            x_tmp = x_tmp.permute(1, 2, 0)

            # Convolution is performed on the last axis of the input data 
            x_tmp = self.tconv.forward(x_tmp, use_attn)

            # Reshape back to original input shape
            x_tmp = x_tmp.permute(2, 0, 1)
            x_tmp = x_tmp.reshape(num_nodes_batch, num_ori, num_channels)
            x_conv.append(x_tmp)

            # Update frame indexing
            start_idx += n_frames*self.num_land_marks
        
        x = torch.cat(x_conv, dim=0)
        
        return x

class TCNUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        use_drop=True,
        dropout_rate=0.1,
        num_points=25,
        padding = 0,
        block_size=41,
        
    ):
        super(TCNUnit, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_channels)
        self.conv_init(self.conv)
        self.in_channels = in_channels
        self.out_channels = out_channels

        #self.attention = nn.Conv1d(out_channels, 1, kernel_size=kernel_size, padding=pad)
        #nn.Conv1d(out_channels, kernel_size=kernel_size, padding = pad)  # Produces a single attention score per time step
        #nn.init.xavier_uniform_(self.attention.weight)
        #nn.init.constant_(self.attention.weight, 0)
        #nn.init.zeros_(self.attention.bias)
        #self.dropout = nn.Dropout(dropout_rate)
        #self.bn = nn.BatchNorm1d(out_channels)
        #bn_init(self.bn, 1)
        self.attention = TAttnUnit(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

    
    def forward(self, x, use_attn = False):
        #if use_attn:
        #    x = self.attention.forward(x)
        y = self.conv(x)
        #y = self.dropout(y)
        
        # Use an activation function 
        y = self.relu(y)

        # Residual connection 
        return y + x


class TAttnUnit(nn.Module):
    def __init__(
        self,
        hid_dim
    ):
        super(TAttnUnit, self).__init__()
        self.hidden_dim = 32
        # Define linear transformations 
        # for Q, K, and V
        self.q_transform = nn.Linear( self.hidden_dim,  self.hidden_dim, bias=False)
        self.k_transform = nn.Linear( self.hidden_dim,  self.hidden_dim, bias=False)
        self.v_transform = nn.Linear( self.hidden_dim,  self.hidden_dim, bias=False)

        
        # Temporal encoding
        self.max_time_steps = 128

        # Not sure about this?
        #self.temporal_encoding = nn.Parameter(torch.randn(1, self.max_time_steps, hid_dim))
    
    def conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

                
    def invariant(self, x):
        return x
    
    def inv_emb_to_q(self, inv):
        q = inv
        return self.q_transform(q)
    
    def inv_emb_to_k(self, inv):
        k = inv
        return self.k_transform(k)
    
    def x_to_v(self, x):
        # Interpreting x as values (appearances, high freq info)
        return self.v_transform(x)

    def forward(self, x):
        # x in : [num_landmarks x num_orientations, num_channels, num_frames]

        # x: [num_landmarks x num_orientations, num_frames, num_channels]
        x = x.permute(0, 2, 1) 
        
        # q, v, k: [num_landmarks x num_orientations, num_frames, num_channels]
        q = self.q_transform(x)
        k = self.k_transform(x)
        v = self.v_transform(x)
        
        
        d_k = q.size(1)

        # scores: [num_landmarks x num_orientations, num_frames x num_frames]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  
        attn_probs = F.softmax(scores, dim=-1)
       
       # y: [num_landmarks x num_orientations, num_frames, hid_dim]
        y = torch.matmul(attn_probs, v)  
        
        # y: [num_landmarks x num_orientations, num_channels, num_frames]
        y = y.permute(0, 2, 1)  
       
        return y
    
    def time_encoding(self, x):
        time_steps = x.size(1)
        temporal_encoding = self.temporal_encoding[:,:time_steps,:]
        x = x + temporal_encoding
        return x 