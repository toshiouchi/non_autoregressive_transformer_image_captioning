import math
import torch
from torch import nn
from torchvision import models


class CNNEncoder(nn.Module):
    '''
    Transformer captioningのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2")

        # 特徴抽出器として使うため全結合層を削除
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        self.linear = nn.Linear(resnet.fc.in_features, dim_embedding)

    '''
    エンコーダの順伝播
    imgs: 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            features = self.backbone(imgs)
            features = features.flatten(1)

        # 全結合
        features = self.linear(features)

        return features
        
class CNNEncoder2(nn.Module):
    '''
    Show and tellのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int):
    #def __init__(self):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2")
        modules = list(resnet.children())[:-4]
        #self.cnnencoder3 = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        #self.linear = nn.Linear(resnet.fc.in_features, dim_embedding)
        in_features = torch.tensor( ( 224 / 8 ) ** 2 ).to( torch.int16 )
        self.linear = nn.Linear( in_features, dim_embedding)

        
    '''
    エンコーダの順伝播
    imgs: 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            features = self.backbone(imgs)
            features = features.flatten(2)
            #features = self.resnet( imgs )

            
        #print( "0 size of features:", features.size())
        # 全結合
        features = self.linear(features)

        return features
        
class CNNEncoder3(nn.Module):
    '''
    Show and tellのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int):
    #def __init__(self):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2")
        modules = list(resnet.children())[:-5]
        #self.cnnencoder3 = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        #self.linear = nn.Linear(resnet.fc.in_features, dim_embedding)
        in_features = torch.tensor( ( 224 / 4 ) ** 2 ).to( torch.int16 )
        self.linear = nn.Linear( in_features, dim_embedding)

        
    '''
    エンコーダの順伝播
    imgs: 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            features = self.backbone(imgs)
            features = features.flatten(2)
            #features = self.resnet( imgs )

            
        #print( "0 size of features:", features.size())
        # 全結合
        features = self.linear(features)

        return features

class PositionalEncoding:
    '''
    位置エンコーディング生成クラス
    eps        : 0で割るのを防ぐための小さい定数
    temperature: 温度定数
    '''
    def __init__(self, eps: float=1e-6, temperature: int=10000):
        self.eps = eps
        self.temperature = temperature

    '''
    位置エンコーディングを生成する関数
    x   : 特徴マップ, [バッチサイズ, チャネル数, 高さ, 幅]
    mask: 画像領域を表すマスク, [バッチサイズ, 高さ, 幅]
    '''
    @torch.no_grad()
    def generate(self, x: torch.Tensor, mask: torch.Tensor):
        # 位置エンコーディングのチャネル数は入力の半分として
        # x方向のエンコーディングとy方向のエンコーディングを用意し、
        # それらを連結することで入力のチャネル数に合わせる
        num_pos_channels = x.shape[1] // 2

        # 温度定数の指数を計算するため、2の倍数を用意
        dim_t = torch.arange(0, num_pos_channels, 2,
                             dtype=x.dtype, device=x.device)
        # sinとcosを計算するために値を複製
        # [0, 2, ...] -> [0, 0, 2, 2, ...]
        dim_t = dim_t.repeat_interleave(2)
        # sinとcosへの入力のの分母となるT^{2i / d}を計算
        dim_t /= num_pos_channels
        dim_t = self.temperature ** dim_t

        # マスクされていない領域の座標を計算
        inverted_mask = ~mask
        y_encoding = inverted_mask.cumsum(1, dtype=torch.float32)
        x_encoding = inverted_mask.cumsum(2, dtype=torch.float32)

        # 座標を0-1に正規化して2πをかける
        y_encoding = 2 * math.pi * y_encoding / \
            (y_encoding.max(dim=1, keepdim=True)[0] + self.eps)
        x_encoding = 2 * math.pi * x_encoding / \
            (x_encoding.max(dim=2, keepdim=True)[0] + self.eps)

        # 座標を保持するテンソルにチャネル軸を追加して、
        # チャネル軸方向にdim_tで割る
        # 偶数チャネルはsin、奇数チャネルはcosの位置エンコーディング
        y_encoding = y_encoding.unsqueeze(1) / \
            dim_t.view(num_pos_channels, 1, 1)
        y_encoding[:, ::2] = y_encoding[:, ::2].sin()
        y_encoding[:, 1::2] = y_encoding[:, 1::2].cos()
        x_encoding = x_encoding.unsqueeze(1) / \
            dim_t.view(num_pos_channels, 1, 1)
        x_encoding[:, ::2] = x_encoding[:, ::2].sin()
        x_encoding[:, 1::2] = x_encoding[:, 1::2].cos()

        encoding = torch.cat((y_encoding, x_encoding), dim=1)

        return encoding
        
class CNNEncoder4(nn.Module):
    '''
    Show and tellのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int):
    #def __init__(self):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2")
        modules = list(resnet.children())[:-5]
        #self.cnnencoder3 = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        #self.linear = nn.Linear(resnet.fc.in_features, dim_embedding)
        in_features = torch.tensor( ( 224 / 4 ) ** 2 ).to( torch.int16 )
        self.linear = nn.Linear( in_features, dim_embedding)

        self.positional_encoding = PositionalEncoding()
        
    '''
    エンコーダの順伝播
    imgs: 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            features = self.backbone(imgs)
            mask = torch.zeros( features.size(0), 56, 56, device = features.device ).to( torch.bool )
            positions = self.positional_encoding.generate( features, mask )
            #print( positions )
            features = features + positions
            features = features.flatten(2)
            #features = self.resnet( imgs )

            
        #print( "0 size of features:", features.size())
        # 全結合
        features = self.linear(features)

        return features
        
class CNNEncoder5(nn.Module):
    '''
    Show and tellのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int):
    #def __init__(self):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2")
        modules = list(resnet.children())[:-4]
        #self.cnnencoder3 = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        #self.linear = nn.Linear(resnet.fc.in_features, dim_embedding)
        in_features = torch.tensor( ( 224 / 8 ) ** 2 ).to( torch.int32 )
        #print( "in_features:", in_features )
        in_features2 = ( 512 * in_features ).to( torch.int32 )
        #print( "in_features2:", in_features2 )
        self.linear1 = nn.Linear( in_features, dim_embedding)
        self.linear2 = nn.Linear( in_features2, dim_embedding )
        
    '''
    エンコーダの順伝播
    imgs: 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            features = self.backbone(imgs)
            features = features.flatten(2)
            features2 = features.flatten(1)
            #print( "size of features2:", features2.size() )
            #features = self.resnet( imgs )

            
        #print( "0 size of features:", features.size())
        # 全結合
        features = self.linear1(features)
        features2 = self.linear2(features2).unsqueeze( 1 )

        return features, features2
