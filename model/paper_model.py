import torch
import torch.nn as nn


# =========================================================
# 1. 通道对齐
# =========================================================

class PaperAlign(nn.Module):
    """
    把残差连接的输入通道数调整为目标通道数。

    c_in > c_out：
    使用 1×1 卷积压缩。

    c_in < c_out：
    在通道方向补 0。

    c_in == c_out：
    保持不变。
    """

    def __init__(self, c_in, c_out):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        if c_in > c_out:
            # 原作者只有权重，没有额外 bias
            self.projection = nn.Conv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(1, 1),
                bias=False,
            )

            nn.init.xavier_uniform_(
                self.projection.weight
            )
        else:
            self.projection = None

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.projection(x)

        if self.c_in < self.c_out:
            padding = x.new_zeros(
                (
                    x.shape[0],
                    self.c_out - self.c_in,
                    x.shape[2],
                    x.shape[3],
                )
            )

            return torch.cat(
                [x, padding],
                dim=1,
            )

        return x


# =========================================================
# 2. 论文时间卷积
# =========================================================

class PaperTemporalConv(nn.Module):
    """
    原作者使用的时间卷积。

    输入：
    [batch, channel, time, node]

    支持：
    - glu
    - relu
    - sigmoid
    - linear
    """

    def __init__(
        self,
        kernel_size,
        c_in,
        c_out,
        activation,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.c_out = c_out
        self.activation = activation

        self.align = PaperAlign(
            c_in,
            c_out,
        )

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

        # 更接近 TensorFlow 1 默认的 Glorot 初始化
        nn.init.xavier_uniform_(
            self.conv.weight
        )

        nn.init.zeros_(
            self.conv.bias
        )

    def forward(self, x):
        # VALID 时间卷积后，残差也要删除前 K-1 个位置
        residual = self.align(x)[
            :,
            :,
            self.kernel_size - 1:,
            :,
        ]

        conv_result = self.conv(x)

        if self.activation == "glu":
            content = conv_result[
                :,
                :self.c_out,
                :,
                :,
            ]

            gate = conv_result[
                :,
                self.c_out:,
                :,
                :,
            ]

            return (
                content + residual
            ) * torch.sigmoid(gate)

        if self.activation == "relu":
            return torch.relu(
                conv_result + residual
            )

        if self.activation == "sigmoid":
            # 原作者 sigmoid 分支不加入残差
            return torch.sigmoid(
                conv_result
            )

        if self.activation == "linear":
            return conv_result

        raise ValueError(
            f"不支持的时间卷积激活函数："
            f"{self.activation}"
        )


# =========================================================
# 3. Chebyshev 图卷积
# =========================================================

class PaperChebGraphConv(nn.Module):
    """
    Chebyshev 多项式图卷积。
    """

    def __init__(
        self,
        c_in,
        c_out,
        Ks,
        gso,
    ):
        super().__init__()

        self.Ks = Ks

        # 将 GSO 注册为模型的一部分，
        # 但它不是需要训练的参数
        self.register_buffer(
            "gso",
            gso.detach().clone(),
        )

        self.weight = nn.Parameter(
            torch.empty(
                Ks,
                c_in,
                c_out,
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(c_out)
        )

        # 等价地把 [Ks, c_in, c_out]
        # 看成 [Ks*c_in, c_out] 初始化
        nn.init.xavier_uniform_(
            self.weight.view(
                Ks * c_in,
                c_out,
            )
        )

    def forward(self, x):
        # [B, C, T, N]
        # →
        # [B, T, N, C]
        x = x.permute(
            0,
            2,
            3,
            1,
        )

        cheb_terms = [x]

        if self.Ks >= 2:
            first_order = torch.einsum(
                "hi,btij->bthj",
                self.gso,
                x,
            )

            cheb_terms.append(
                first_order
            )

        for _ in range(2, self.Ks):
            next_order = (
                torch.einsum(
                    "hi,btij->bthj",
                    2 * self.gso,
                    cheb_terms[-1],
                )
                - cheb_terms[-2]
            )

            cheb_terms.append(
                next_order
            )

        # [B, T, Ks, N, C_in]
        cheb_features = torch.stack(
            cheb_terms,
            dim=2,
        )

        # [B, T, N, C_out]
        output = torch.einsum(
            "btkni,kio->btno",
            cheb_features,
            self.weight,
        )

        return output + self.bias


# =========================================================
# 4. 论文空间图卷积层
# =========================================================

class PaperSpatialConv(nn.Module):
    """
    图卷积结果与残差相加，然后使用 ReLU。
    """

    def __init__(
        self,
        Ks,
        c_in,
        c_out,
        gso,
    ):
        super().__init__()

        self.align = PaperAlign(
            c_in,
            c_out,
        )

        self.graph_conv = PaperChebGraphConv(
            c_in=c_in,
            c_out=c_out,
            Ks=Ks,
            gso=gso,
        )

    def forward(self, x):
        residual = self.align(x)

        graph_result = self.graph_conv(
            x
        ).permute(
            0,
            3,
            1,
            2,
        )

        return torch.relu(
            graph_result + residual
        )


# =========================================================
# 5. 原作者 ST-Conv Block
# =========================================================

class PaperSTConvBlock(nn.Module):
    """
    原作者 ST-Conv Block：

    Temporal GLU
    → Spatial Graph Conv + ReLU
    → Temporal ReLU
    → LayerNorm
    → Dropout
    """

    def __init__(
        self,
        Kt,
        Ks,
        n_vertex,
        c_in,
        c_temporal,
        c_out,
        gso,
        droprate=0.0,
    ):
        super().__init__()

        # 第一次时间卷积：GLU
        self.temporal1 = PaperTemporalConv(
            kernel_size=Kt,
            c_in=c_in,
            c_out=c_temporal,
            activation="glu",
        )

        # 图卷积：ReLU
        self.spatial = PaperSpatialConv(
            Ks=Ks,
            c_in=c_temporal,
            c_out=c_temporal,
            gso=gso,
        )

        # 第二次时间卷积：ReLU
        self.temporal2 = PaperTemporalConv(
            kernel_size=Kt,
            c_in=c_temporal,
            c_out=c_out,
            activation="relu",
        )

        # 原作者 epsilon 为 1e-6
        self.norm = nn.LayerNorm(
            [n_vertex, c_out],
            eps=1e-6,
        )

        self.dropout = nn.Dropout(
            p=droprate
        )

    def forward(self, x):
        x = self.temporal1(x)
        x = self.spatial(x)
        x = self.temporal2(x)

        # [B, C, T, N]
        # →
        # [B, T, N, C]
        x = self.norm(
            x.permute(
                0,
                2,
                3,
                1,
            )
        )

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        return self.dropout(x)


# =========================================================
# 6. 原作者 Output Layer
# =========================================================

class PaperOutputBlock(nn.Module):
    """
    原作者输出层：

    Temporal GLU
    → LayerNorm
    → kernel=1 Temporal Sigmoid
    → 共享全连接
    → 每个节点独立 bias
    """

    def __init__(
        self,
        Ko,
        n_vertex,
        channels,
    ):
        super().__init__()

        self.temporal1 = PaperTemporalConv(
            kernel_size=Ko,
            c_in=channels,
            c_out=channels,
            activation="glu",
        )

        self.norm = nn.LayerNorm(
            [n_vertex, channels],
            eps=1e-6,
        )

        self.temporal2 = PaperTemporalConv(
            kernel_size=1,
            c_in=channels,
            c_out=channels,
            activation="sigmoid",
        )

        # 原作者共享一个 channel → 1 的映射
        self.fully_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=(1, 1),
            bias=False,
        )

        nn.init.xavier_uniform_(
            self.fully_conv.weight
        )

        # 原作者为每个道路节点保存独立 bias
        self.node_bias = nn.Parameter(
            torch.zeros(
                1,
                1,
                1,
                n_vertex,
            )
        )

    def forward(self, x):
        x = self.temporal1(x)

        x = self.norm(
            x.permute(
                0,
                2,
                3,
                1,
            )
        ).permute(
            0,
            3,
            1,
            2,
        )

        x = self.temporal2(x)

        return (
            self.fully_conv(x)
            + self.node_bias
        )


# =========================================================
# 7. 完整论文 STGCN
# =========================================================

class PaperSTGCN(nn.Module):
    """
    论文对齐的两层 STGCN。

    Block 1：
    1 → 32 → 32 → 64

    Block 2：
    64 → 32 → 32 → 128
    """

    def __init__(
        self,
        Kt,
        Ks,
        n_his,
        n_vertex,
        gso,
        droprate=0.0,
    ):
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

        self.Ko = (
            n_his
            - 2
            * 2
            * (Kt - 1)
        )

        if self.Ko <= 1:
            raise ValueError(
                "Output Layer 的 Ko 必须大于 1，"
                f"当前 Ko={self.Ko}"
            )

        self.output = PaperOutputBlock(
            Ko=self.Ko,
            n_vertex=n_vertex,
            channels=128,
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.output(x)

        return x