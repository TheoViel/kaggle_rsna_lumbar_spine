import timm
import torch
import warnings
import torch.nn as nn

from model_zoo.layers import GeM, Attention

warnings.simplefilter(action="ignore", category=UserWarning)


def define_model(
    name,
    num_classes=2,
    num_classes_aux=0,
    n_channels=3,
    pretrained_weights="",
    pretrained=True,
    reduce_stride=False,
    increase_stride=False,
    drop_rate=0,
    drop_path_rate=0,
    pooling="avg",
    head_3d="",
    delta=2,
    n_frames=1,
    verbose=1,
):
    """
    Define a classification model with a specified encoder.

    Args:
        name (str): The name of the encoder model.
        num_classes (int, optional): The number of primary target classes. Defaults to 2.
        num_classes_aux (int, optional): The number of auxiliary target classes. Defaults to 0.
        n_channels (int, optional): The number of input channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pretrained encoder weights. Defaults to "".
        pretrained (bool, optional): Whether to use pretrained encoder weights. Defaults to True.
        reduce_stride (bool, optional): Whether to reduce the encoder stride. Defaults to False.
        increase_stride (bool, optional): Whether to increase the encoder stride. Defaults to False.
        drop_rate (float, optional): Dropout rate. Defaults to 0.
        drop_path_rate (float, optional): Drop path rate. Defaults to 0.
        pooling (str, optional): Pooling method. Defaults to "avg".
        head_3d (str, optional): 3D head method. Defaults to "".
        delta (int, optional): Delta value. Defaults to 2.
        n_frames (int, optional): Number of frames. Defaults to 1.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        nn.Module: The defined classification model.
    """
    if drop_path_rate > 0 and "coat_" not in name:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            global_pool="",
        )
    elif "efficientvit" in name:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
        )
        encoder.num_features = encoder.head.classifier[0].out_features
    elif "eva" in name:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
        )
    elif "coat_" in name:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            global_pool=pooling if pooling in ["avg", "token"] else "avg",
        )
        if pooling == "flatten":
            encoder.forward = lambda x: encoder.forward_features(x)[:, 1:]
    else:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
    encoder.name = name

    model = ClsModel(
        encoder,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
        drop_rate=drop_rate,
        pooling=pooling,
        head_3d=head_3d,
        n_frames=n_frames,
        delta=delta,
    )
    model.name = name

    if pretrained_weights:
        import re
        if verbose:
            print(f'-> Loading encoder weights from {pretrained_weights}\n')
        sd = torch.load(pretrained_weights)
        d = {}
        for k in sd:
            if "model.encoder.model" in k:
                k_ = re.sub("model.encoder.model.", "", k)
                k_ = re.sub("stages_", "stages.", k_)
            d[k_] = sd[k]
        del d["stages.3.blocks.1.mlp.fc2.bias"]
        model.encoder.load_state_dict(d, strict=False)

    if reduce_stride:
        model.reduce_stride()
    if increase_stride:
        model.increase_stride()

    return model


class ClsModel(nn.Module):
    """
    A classification model with an encoder, pooling, and optional 3D head.
    """
    def __init__(
        self,
        encoder,
        num_classes=2,
        num_classes_aux=0,
        n_channels=3,
        drop_rate=0,
        pooling="avg",
        head_3d="",
        n_frames=1,
        delta=2,
    ):
        """
        Constructor for the classification model.

        Args:
            encoder (nn.Module): The encoder model.
            num_classes (int, optional): The number of primary target classes. Defaults to 2.
            num_classes_aux (int, optional): The number of auxiliary target classes. Defaults to 0.
            n_channels (int, optional): The number of input channels. Defaults to 3.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            pooling (str, optional): Pooling method. Defaults to "avg".
            head_3d (str, optional): 3D head method. Defaults to "".
            n_frames (int, optional): Number of frames. Defaults to 1.
            delta (int, optional): Delta value. Defaults to 2.
        """
        super().__init__()

        self.encoder = encoder
        self.nb_ft = encoder.num_features

        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.n_channels = n_channels
        self.pooling = pooling
        self.head_3d = head_3d
        self.delta = delta

        self.dense = nn.Identity()
        if pooling == "gem":
            self.global_pool = GeM(p_trainable=False)
        elif pooling in ["avg_h", "avg_w"]:
            if "224" in encoder.name or encoder.name == "coat_lite_medium":
                self.dense = nn.Sequential(
                    nn.Linear(self.nb_ft * 7, self.nb_ft)
                )
            elif "384" in encoder.name:
                self.dense = nn.Sequential(
                    nn.Linear(self.nb_ft * 12, self.nb_ft)
                )
            else:
                raise NotImplementedError
        elif pooling == "flatten":
            if "224" in encoder.name or encoder.name == "coat_lite_medium":
                self.nb_ft *= 49  # coat
            elif "384" in encoder.name:
                self.nb_ft *= 144
            else:
                raise NotImplementedError

        self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()

        # 3D Head
        if head_3d == "lstm":
            self.lstm = nn.LSTM(
                self.nb_ft, self.nb_ft // 4, batch_first=True, bidirectional=True
            )
        elif head_3d == "lstm_side":
            self.lstm_center = nn.LSTM(
                self.nb_ft, self.nb_ft // 4, batch_first=True, bidirectional=True
            )
            self.lstm_left = nn.LSTM(
                self.nb_ft, self.nb_ft // 4, batch_first=True, bidirectional=True
            )
            self.lstm_right = nn.LSTM(
                self.nb_ft, self.nb_ft // 4, batch_first=True, bidirectional=True
            )
        elif head_3d == "lstm_att":
            self.lstm = nn.LSTM(
                self.nb_ft, self.nb_ft // 2, batch_first=True, bidirectional=True
            )
            self.att = Attention(self.nb_ft, self.nb_ft)
        elif head_3d == "transfo":
            self.transfo = nn.TransformerEncoderLayer(
                self.nb_ft,
                8,
                dim_feedforward=self.nb_ft * 2,
                dropout=0.1,
                activation=nn.Mish(),
                batch_first=True,
            )

        # Logits
        if head_3d == "lstm_side":
            self.logits_center = nn.Linear(self.nb_ft, 3)
            self.logits_left = nn.Linear(self.nb_ft, 6)
            self.logits_right = nn.Linear(self.nb_ft, 6)
        else:
            self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        self._update_num_channels()

    def _update_num_channels(self):
        """
        Update the number of input channels for the encoder.
        """
        if self.n_channels != 3:
            if "convnext" in self.encoder.name:
                conv = self.encoder.stem[0]
            elif "coat_lite" in self.encoder.name:
                conv = self.encoder.patch_embed1.proj
            elif "coatnet" in self.encoder.name:
                conv = self.encoder.stem.conv1

            new_conv = nn.Conv2d(
                self.n_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
            )

            new_conv_w = new_conv.weight.clone().detach()
            new_conv_w[:, :3] = conv.weight.clone().detach()
            new_conv.weight = torch.nn.Parameter(new_conv_w, requires_grad=True)

            if conv.bias is not None:
                new_conv_b = conv.bias.clone().detach()
                new_conv.bias = torch.nn.Parameter(new_conv_b, requires_grad=True)

            if "convnext" in self.encoder.name:
                self.encoder.stem[0] = new_conv
            elif "coat_lite" in self.encoder.name:
                self.encoder.patch_embed1.proj = new_conv
            elif "coatnet" in self.encoder.name:
                self.encoder.stem.conv1 = new_conv

    def reduce_stride(self):
        """
        Reduce the stride of the first layer of the encoder.
        """
        if "efficient" in self.encoder.name:
            self.encoder.conv_stem.stride = (1, 1)
        elif "nfnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (1, 1)
        else:
            raise NotImplementedError

    def increase_stride(self):
        """
        Increase the stride of the first layer of the encoder.
        """
        if "efficient" in self.encoder.name:
            self.encoder.conv_stem.stride = (4, 4)
        elif "nfnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (4, 4)
        else:
            raise NotImplementedError

    def extract_features(self, x):
        """
        Extract features from input images.

        Args:
            x (torch.Tensor): Input images of shape [batch_size x n_channels x H x W].

        Returns:
            torch.Tensor: Extracted features of shape [batch_size x num_features].
        """
        fts = self.encoder(x)
        # print(fts.size())

        # Reorder features for transformers
        if "vit" in self.name:
            fts = fts.transpose(-1, -2)
        elif "swin" in self.name:
            fts = fts.transpose(-1, -2).transpose(-2, -3)

        # Pool
        if self.pooling == "gem":
            assert len(fts.size()) == 4
            fts = self.global_pool(fts)[:, :, 0, 0]
        elif self.pooling == "avg":
            while len(fts.size()) > 2:
                fts = fts.mean(-1)
        elif self.pooling == "avg_w":  # For L1 / L2 / ...
            while len(fts.size()) > 3:
                fts = fts.mean(-1)
            fts = fts.flatten(-2, -1)
        elif self.pooling == "avg_h":  # For left / right
            while len(fts.size()) > 3:
                fts = fts.mean(-2)
            fts = fts.flatten(-2, -1)
        elif self.pooling == "flatten":
            while len(fts.size()) > 2:
                fts = fts.flatten(-2, -1)

        # print(fts.size())
        fts = self.dense(fts)
        # print(fts.size())

        fts = self.dropout(fts)

        return fts

    def get_logits(self, fts):
        """
        Compute logits for the primary and auxiliary classes.

        Args:
            fts (torch.Tensor): Features of shape [batch_size x num_features].

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
            torch.Tensor: Logits for the auxiliary classes of shape [batch_size x num_classes_aux].
        """
        logits = self.logits(fts)

        if self.num_classes_aux:
            logits_aux = self.logits_aux(fts)
        else:
            logits_aux = torch.zeros((fts.size(0)))

        return logits, logits_aux

    def forward_side(self, x):
        """
        Forward function for the lstm_side 3D head.

        Args:
            x (torch.Tensor [batch_size x n_frames x num_features]): Input features for the 3D head.

        Returns:
            torch.Tensor: Result of the 3D head.
        """
        mid = x.size(1) // 2
        try:
            delta = self.delta
        except Exception:
            delta = 2  # mid // 2
        x_center, _ = self.lstm_center(x[:, mid - delta: mid + delta + 1])
        x_center = torch.cat([x_center.mean(1), x_center.amax(1)], -1)

        x_left, _ = self.lstm_left(x[:, mid:])
        x_left = torch.cat([x_left.mean(1), x_left.amax(1)], -1)

        x_right, _ = self.lstm_right(x[:, :mid + 1])
        x_right = torch.cat([x_right.mean(1), x_right.amax(1)], -1)

        logits_center = self.logits_center(x_center)
        logits_left = self.logits_left(x_left)
        logits_right = self.logits_right(x_right)

        logits = torch.zeros(x.size(0), self.num_classes).to(x.device)

        logits[:, :3] = logits_center  # scs
        logits[:, 3:6] = logits_left[:, :3]  # nfn
        logits[:, 6:9] = logits_right[:, :3]  # nfn
        logits[:, 9:12] = logits_left[:, 3:]  # ss
        logits[:, 12:] = logits_right[:, 3:]  # ss

        return logits, torch.zeros((x.size(0)))

    def forward_head_3d(self, x):
        """
        Forward function for the 3D head.

        Args:
            x (torch.Tensor [batch_size x n_frames x num_features]): Input features for the 3D head.

        Returns:
            torch.Tensor: Result of the 3D head.
        """
        if self.head_3d == "avg":
            return x.mean(1)

        elif self.head_3d == "max":
            return x.amax(1)

        elif self.head_3d == "lstm":
            x, _ = self.lstm(x)
            mean = x.mean(1)
            max_ = x.amax(1)
            x = torch.cat([mean, max_], -1)

        elif self.head_3d == "lstm_att":
            x, _ = self.lstm(x)
            x = self.att(x)

        elif self.head_3d == "transfo":
            x = self.transfo(x).mean(1)

        return x

    def forward(self, x, return_fts=False):
        """
        Forward function for the model.

        Args:
            x (torch.Tensor): Input images of shape [batch_size (x n_frames) x n_channels x H x W].
            return_fts (bool): Flag to return features.

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
            torch.Tensor: Logits for the auxiliary classes of shape [batch_size x num_classes_aux].
        """
        if self.head_3d:
            bs, n_frames, c, h, w = x.size()
            x = x.view(bs * n_frames, c, h, w)

        fts = self.extract_features(x)

        if self.head_3d == "lstm_side":
            fts = fts.contiguous().view(bs, n_frames, -1)
            return self.forward_side(fts)

        if self.head_3d:
            fts = fts.contiguous().view(bs, n_frames, -1)
            fts = self.forward_head_3d(fts)

        logits, logits_aux = self.get_logits(fts)

        if return_fts:
            return logits, fts
        return logits, logits_aux
