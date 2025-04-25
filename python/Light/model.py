import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NUM_ACTIONS
# ==============================
# ネットワーク定義
# ==============================
# ResidualBlock : 標準的な残差ブロック（スキップ接続を含む）
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        パラメータ:
          channels (int): 入力および出力のチャネル数
        """
        super(ResidualBlock, self).__init__()
        # カーネルサイズ3、パディング1の畳み込み層を2層定義
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)  # 畳み込み後のバッチ正規化層
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        # 入力を残差として保持（スキップ接続）
        residual = x
        # 1回目の畳み込み→正規化→ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 2回目の畳み込み→正規化
        out = self.bn2(self.conv2(out))
        # スキップ接続により、入力と出力を加算
        out += residual
        # 最終的にReLUを適用して出力
        return F.relu(out)

# DilatedResidualBlock : 拡張（dilated）畳み込みを用いた残差ブロック
class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=2):
        """
        パラメータ:
          channels (int): 入力および出力チャネル数
          dilation (int): 拡張率（dilation rate）
        """
        super(DilatedResidualBlock, self).__init__()
        # 拡張畳み込み（dilation指定、パディングにdilationの値を利用）
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        # 通常の畳み込み層
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x  # 入力をスキップ接続用に保持
        out = F.relu(self.bn1(self.conv1(x)))  # 拡張畳み込み＋正規化＋ReLU
        out = self.bn2(self.conv2(out))         # 通常の畳み込み＋正規化
        out += residual                         # スキップ接続：入力を加算
        return F.relu(out)                      # 最後にReLUを適用して出力

# SelfAttention : セルフアテンション機構の層。空間的な自己相関を計算する
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        """
        パラメータ:
          in_channels (int): 入力チャネル数
        """
        super(SelfAttention, self).__init__()
        # クエリを得るための1x1畳み込み。チャネル数を1/8に縮小
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # キーを得るための1x1畳み込み（同じくチャネルを1/8に）
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # バリューを得るための1x1畳み込み（チャネル数は変えない）
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 出力へのスケールパラメータ（学習可能なパラメータとして0で初期化）
        self.gamma = nn.Parameter(torch.zeros(1))
        # ソフトマックス関数でアテンション係数を正規化（最後の次元で適用）
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        batch, C, H, W = x.size()  # 入力の形状を取得
        # クエリを計算し、形状を (batch, H*W, C//8) に変形
        proj_query = self.query_conv(x).view(batch, -1, H * W).permute(0, 2, 1)
        # キーを計算し、形状を (batch, C//8, H*W) に変形
        proj_key = self.key_conv(x).view(batch, -1, H * W)
        # クエリとキーのバッチ積により相関（エネルギー）を計算
        energy = torch.bmm(proj_query, proj_key)
        # ソフトマックスにより、各位置での注意重みを計算
        attention = self.softmax(energy)
        # バリューを計算し、形状を (batch, C, H*W) に変形
        proj_value = self.value_conv(x).view(batch, -1, H * W)
        # バッチ積により、注目情報を反映した出力を計算（注意の転置を使用）
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)  # 元の空間サイズに戻す
        # スケールパラメータを掛けた出力を元入力に足し込み、最終出力とする
        return self.gamma * out + x

# EnhancedResNetPolicyValueNetwork : 改良版残差ネットワークに基づく、ポリシーとバリューを同時に出力するネットワーク
class EnhancedResNetPolicyValueNetwork(nn.Module):
    def __init__(self, board_size, num_channels, num_residual_blocks, in_channels):
        """
        パラメータ:
          board_size (int): 盤面のサイズ（例：19）
          num_channels (int): 入力特徴マップのチャネル数
          num_residual_blocks (int): 残差ブロックの数
          in_channels (int): ネットワークへの入力チャネル数
        """
        super(EnhancedResNetPolicyValueNetwork, self).__init__()
        self.board_size = board_size
        # 入力層：3x3畳み込みとバッチ正規化を適用して特徴抽出
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        # 残差ブロックをリストに格納。一定ごとに拡張畳み込みブロックを使用
        blocks = []
        for i in range(num_residual_blocks):
            if i % 4 == 0:
                blocks.append(DilatedResidualBlock(num_channels, dilation=2))
            else:
                blocks.append(ResidualBlock(num_channels))
        # リストのブロックを連結してシーケンシャルなモデルにする
        self.residual_blocks = nn.Sequential(*blocks)
        # セルフアテンション層を配置（特徴マップ内の依存関係を補正）
        self.attention = SelfAttention(num_channels)
        # ----- ポリシーヘッドの構築 -----
        # 1x1畳み込みとバッチ正規化でチャネル数を2に縮小
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.dropout_policy = nn.Dropout(p=0.5)  # 過学習防止のためDropoutを適用
        # 全結合層：2チャネルの各位置をフラット化し、全ての行動（盤面のマス＋パス）を出力
        self.fc_policy = nn.Linear(2 * board_size * board_size, NUM_ACTIONS)
        # ----- バリューヘッドの構築 -----
        # 1x1畳み込みとバッチ正規化で1チャネルに縮小し、盤面全体の特徴を抽出
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        # 隠れ層全結合：盤面情報を64次元へ変換
        self.fc_value1 = nn.Linear(board_size * board_size, 64)
        self.dropout_value = nn.Dropout(p=0.5)
        # 最終出力は2ユニット。1つはvalue（勝ち負けの評価）、もう1つはmargin（差分情報）として出力
        self.fc_value2 = nn.Linear(64, 2)
    def forward(self, x):
        # 入力層処理：畳み込み→正規化→ReLU
        x = F.relu(self.bn_input(self.conv_input(x)))
        # 複数の残差ブロック（および拡張畳み込みブロック）の適用
        x = self.residual_blocks(x)
        # セルフアテンション層を適用して特徴マップを補正
        x = self.attention(x)
        # ---- ポリシーヘッド処理 ----
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = self.dropout_policy(p)
        p = p.view(p.size(0), -1)  # フラット化して全結合層への入力形状に
        p = self.fc_policy(p)
        p = F.log_softmax(p, dim=1)  # 出力の対数確率に変換
        # ---- バリューヘッド処理 ----
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = self.dropout_value(v)
        out = self.fc_value2(v)
        # 1ユニット目を [tanh] で正規化し[-1,1]の範囲に収め、残りをmarginとしてそのまま出力
        value = torch.tanh(out[:, 0])
        margin = out[:, 1]
        return p, (value, margin)

