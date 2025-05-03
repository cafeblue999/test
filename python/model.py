import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    チャネル方向の重みづけを学習する Squeeze-and-Excitation ブロック。
    各チャネルの重要度を自己学習し、特徴マップを強調・抑制する。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    通常の ResNet 構造に SEBlock を付加した ResidualBlock。
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class MultiHeadSelfAttention(nn.Module):
    """
    相対位置エンコーディング付きの Multi-Head Self Attention。
    盤面上の位置に応じて注意を変化させることができる。
    """
    def __init__(self, in_channels, heads=4):
        super().__init__()
        assert in_channels % heads == 0
        self.heads = heads
        self.key = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.query = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.scale = (in_channels // heads) ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))
        self.rel_pos_h = nn.Parameter(torch.randn((1, in_channels // heads, 1, 19)), requires_grad=True)
        self.rel_pos_w = nn.Parameter(torch.randn((1, in_channels // heads, 19, 1)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.key(x).reshape(B, self.heads, C // self.heads, H * W)
        q = self.query(x).reshape(B, self.heads, C // self.heads, H * W)
        v = self.value(x).reshape(B, self.heads, C // self.heads, H * W)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale  # (B, heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)

        # 相対位置エンコーディングを各ヘッドごとに繰り返し、チャンネル数 C に揃える
        # rel_pos_h: (1, C//heads,1, W) → (1, C,1, W)
        rel_h = self.rel_pos_h.repeat(1, self.heads, 1, 1)
        rel_w = self.rel_pos_w.repeat(1, self.heads, 1, 1)
        # ブロードキャストして加算
        out = out + rel_h.expand(B, -1, H, W) + rel_w.expand(B, -1, H, W)
        return self.gamma * out + x


class EnhancedResNetPolicyValueNetwork(nn.Module):
    """
    改良版 ResNet ベースの囲碁ポリシー・バリューネットワーク。
    - SEBlock を各 ResidualBlock に統合
    - SelfAttention を後半に分散挿入
    - margin 出力なし（policy, value のみ）
    """
    def __init__(self, board_size=19, in_channels=17, num_channels=256, num_blocks=40):
        super().__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size + 1  # +1 for pass

        # 入力層
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

        # Residual blocks + SelfAttention（後半に集中）
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResidualBlock(num_channels))
            if i >= 30 and (i % 2 == 0):
                blocks.append(MultiHeadSelfAttention(num_channels, heads=4))
        self.residual_blocks = nn.Sequential(*blocks)

        # Policy Head（打ち手の分布予測）
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.num_actions)
        )

        # Value Head（勝率予測）
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()  # 出力を [-1, 1] 範囲に制限
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
