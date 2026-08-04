import torch
import torch.nn as nn


class PaperAlign(nn.Module):
    """Align residual channels by projection, zero-padding, or identity."""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        if c_in > c_out:
            # The TensorFlow reference uses a weight-only 1x1 projection.
            self.projection = nn.Conv2d(
                in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=False
            )

            nn.init.xavier_uniform_(self.projection.weight)
        else:
            self.projection = None

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.projection(x)

        if self.c_in < self.c_out:
            padding = x.new_zeros(
                (x.shape[0], self.c_out - self.c_in, x.shape[2], x.shape[3])
            )

            return torch.cat([x, padding], dim=1)

        return x


class PaperTemporalConv(nn.Module):
    """Temporal convolution matching the reference activation semantics."""

    def __init__(self, kernel_size, c_in, c_out, activation):
        super().__init__()

        self.kernel_size = kernel_size
        self.c_out = c_out
        self.activation = activation

        self.align = PaperAlign(c_in, c_out)

        if activation == "glu":
            conv_out_channels = 2 * c_out
        else:
            conv_out_channels = c_out

        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=conv_out_channels,
            kernel_size=(kernel_size, 1),
            bias=True,
        )

        # TensorFlow 1 uses Glorot initialization for this kernel.
        nn.init.xavier_uniform_(self.conv.weight)

        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # VALID convolution shortens time, so the residual must be cropped too.
        residual = self.align(x)[:, :, self.kernel_size - 1 :, :]

        conv_result = self.conv(x)

        if self.activation == "glu":
            content = conv_result[:, : self.c_out, :, :]

            gate = conv_result[:, self.c_out :, :, :]

            return (content + residual) * torch.sigmoid(gate)

        if self.activation == "relu":
            return torch.relu(conv_result + residual)

        if self.activation == "sigmoid":
            # The reference sigmoid branch intentionally omits the residual.
            return torch.sigmoid(conv_result)

        if self.activation == "linear":
            return conv_result

        raise ValueError(f"Unsupported temporal activation: {self.activation}")


class PaperChebGraphConv(nn.Module):
    """Chebyshev polynomial graph convolution."""

    def __init__(self, c_in, c_out, Ks, gso):
        super().__init__()

        self.Ks = Ks

        # The GSO follows the model across devices but is not trainable.
        self.register_buffer("gso", gso.detach().clone())

        self.weight = nn.Parameter(torch.empty(Ks, c_in, c_out))

        self.bias = nn.Parameter(torch.zeros(c_out))

        # Initialize the stacked [Ks * c_in, c_out] transform as one kernel.
        nn.init.xavier_uniform_(self.weight.view(Ks * c_in, c_out))

    def forward(self, x):
        # [B, C, T, N] -> [B, T, N, C]
        x = x.permute(0, 2, 3, 1)

        cheb_terms = [x]

        if self.Ks >= 2:
            first_order = torch.einsum("hi,btij->bthj", self.gso, x)

            cheb_terms.append(first_order)

        for _ in range(2, self.Ks):
            next_order = (
                torch.einsum("hi,btij->bthj", 2 * self.gso, cheb_terms[-1])
                - cheb_terms[-2]
            )

            cheb_terms.append(next_order)

        # [B, T, Ks, N, C_in]
        cheb_features = torch.stack(cheb_terms, dim=2)

        # [B, T, N, C_out]
        output = torch.einsum("btkni,kio->btno", cheb_features, self.weight)

        return output + self.bias


class PaperSpatialConv(nn.Module):
    """Apply graph convolution, add the aligned residual, and activate."""

    def __init__(self, Ks, c_in, c_out, gso):
        super().__init__()

        self.align = PaperAlign(c_in, c_out)

        self.graph_conv = PaperChebGraphConv(c_in=c_in, c_out=c_out, Ks=Ks, gso=gso)

    def forward(self, x):
        residual = self.align(x)

        graph_result = self.graph_conv(x).permute(0, 3, 1, 2)

        return torch.relu(graph_result + residual)


class PaperSTConvBlock(nn.Module):
    """
    ST-Conv block used by the TensorFlow reference:

    Temporal GLU
    -> spatial graph convolution and ReLU
    -> temporal ReLU
    -> LayerNorm
    -> dropout
    """

    def __init__(self, Kt, Ks, n_vertex, c_in, c_temporal, c_out, gso, droprate=0.0):
        super().__init__()

        self.temporal1 = PaperTemporalConv(
            kernel_size=Kt, c_in=c_in, c_out=c_temporal, activation="glu"
        )

        self.spatial = PaperSpatialConv(
            Ks=Ks, c_in=c_temporal, c_out=c_temporal, gso=gso
        )

        self.temporal2 = PaperTemporalConv(
            kernel_size=Kt, c_in=c_temporal, c_out=c_out, activation="relu"
        )

        # Match the epsilon used by the reference implementation.
        self.norm = nn.LayerNorm([n_vertex, c_out], eps=1e-6)

        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.temporal1(x)
        x = self.spatial(x)
        x = self.temporal2(x)

        # LayerNorm operates on [node, channel] in the reference layout.
        x = self.norm(x.permute(0, 2, 3, 1))

        x = x.permute(0, 3, 1, 2)

        return self.dropout(x)


class PaperOutputBlock(nn.Module):
    """
    Output block used by the TensorFlow reference:

    Temporal GLU
    -> LayerNorm
    -> kernel-1 temporal sigmoid
    -> shared channel projection
    -> node-specific bias
    """

    def __init__(self, Ko, n_vertex, channels):
        super().__init__()

        self.temporal1 = PaperTemporalConv(
            kernel_size=Ko, c_in=channels, c_out=channels, activation="glu"
        )

        self.norm = nn.LayerNorm([n_vertex, channels], eps=1e-6)

        self.temporal2 = PaperTemporalConv(
            kernel_size=1, c_in=channels, c_out=channels, activation="sigmoid"
        )

        # The channel-to-output projection is shared across nodes.
        self.fully_conv = nn.Conv2d(
            in_channels=channels, out_channels=1, kernel_size=(1, 1), bias=False
        )

        nn.init.xavier_uniform_(self.fully_conv.weight)

        # The reference output keeps a separate bias for each sensor.
        self.node_bias = nn.Parameter(torch.zeros(1, 1, 1, n_vertex))

    def forward(self, x):
        x = self.temporal1(x)

        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.temporal2(x)

        return self.fully_conv(x) + self.node_bias


class PaperSTGCN(nn.Module):
    """
    Two-block STGCN with reference channel widths.

    Block 1: 1 -> 32 -> 32 -> 64
    Block 2: 64 -> 32 -> 32 -> 128
    """

    def __init__(self, Kt, Ks, n_his, n_vertex, gso, droprate=0.0):
        super().__init__()

        self.block1 = PaperSTConvBlock(
            Kt=Kt,
            Ks=Ks,
            n_vertex=n_vertex,
            c_in=1,
            c_temporal=32,
            c_out=64,
            gso=gso,
            droprate=droprate,
        )

        self.block2 = PaperSTConvBlock(
            Kt=Kt,
            Ks=Ks,
            n_vertex=n_vertex,
            c_in=64,
            c_temporal=32,
            c_out=128,
            gso=gso,
            droprate=droprate,
        )

        self.Ko = n_his - 2 * 2 * (Kt - 1)

        if self.Ko <= 1:
            raise ValueError(f"Output-layer Ko must be greater than 1, got {self.Ko}")

        self.output = PaperOutputBlock(Ko=self.Ko, n_vertex=n_vertex, channels=128)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.output(x)

        return x
