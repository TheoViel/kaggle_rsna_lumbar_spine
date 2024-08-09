import torch
import torch.nn as nn


def define_model(
    name="rnn_att",
    ft_dim=2048,
    layer_dim=64,
    dense_dim=256,
    resize=15,
    p=0.1,
    num_classes=2,
    num_classes_aux=0,
    n_fts=0,
):
    """
    Define the level 2 model.

    Args:
        name (str): The name of the model to define. Default is "rnn_att".
        ft_dim (int): Dimension of input features. Default is 2048.
        layer_dim (int): Dimension of LSTM layers. Default is 64.
        dense_dim (int): Dimension of the dense layer. Default is 256.
        p (float): Dropout probability. Default is 0.1.
        num_classes (int): Number of main classes. Default is 2.
        num_classes_aux (int): Number of auxiliary classes. Default is 0.
        n_fts (int): Number of features to use. Default is 0.

    Returns:
        nn.Module: The defined model.
    """
    if name == "rnn_att":
        return RNNAttModel(
            ft_dim=ft_dim,
            lstm_dim=layer_dim,
            dense_dim=dense_dim,
            resize=resize,
            p=p,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
        )
    elif name == "baseline":
        return BaselineModel(
            ft_dim=ft_dim,
            lstm_dim=layer_dim,
            dense_dim=dense_dim,
            resize=resize,
            p=p,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
        )
    else:
        raise NotImplementedError


class BaselineModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        lstm_dim=64,
        resize=15,
        dense_dim=64,
        p=0.1,
        num_classes=8,
        num_classes_aux=0,
        n_fts=0,
    ):
        """
        Constructor.

        Args:
            ft_dim (int): The dimension of input features. Defaults to 64.
            lstm_dim (int): The dimension of the LSTM layer. Defaults to 64.
            n_lstm (int): The number of LSTM layers. Defaults to 1.
            dense_dim (int): The dimension of the dense layer. Defaults to 64.
            p (float): Dropout probability. Defaults to 0.1.
            num_classes (int): The number of primary target classes. Defaults to 8.
            num_classes_aux (int): The number of auxiliary target classes. Defaults to 0.
            n_fts (int): The number of additional features. Defaults to 0.

        """
        super().__init__()
        self.n_fts = n_fts
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlps = nn.ModuleDict({
            "nfn": nn.Sequential(nn.Linear(15, dense_dim), nn.Mish()),
            "scs": nn.Sequential(nn.Linear(15, dense_dim), nn.Mish()),
            "ss": nn.Sequential(nn.Linear(6, dense_dim), nn.Mish()),
        })

        self.lstms = nn.ModuleDict({
            k: nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
            for k in ['nfn', 'scs', "ss"]
        })

        self.crop_mlp = nn.Sequential(nn.Linear(45, dense_dim * 4), nn.Mish())

        self.logits = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(2 * lstm_dim * 4 * resize + dense_dim * 4, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(self, x, ft=None):
        """
        Forward pass of the RNN with attention model.
        """
        bs = x['ss'].size(0)

        # x['ss'] = torch.cat([x['ss'], x['ss_aux']], -1)

        fts = {}
        for k in self.mlps:
            features = self.mlps[k](x[k])
            features_lstm, _ = self.lstms[k](features)
            fts[k] = features_lstm.reshape(bs, -1)

        fts = torch.cat([fts[k] for k in fts], -1)

        crop_fts = torch.cat([x['nfn_crop'], x['scs_crop']], 1)
        crop_fts = self.crop_mlp(crop_fts)

        fts = torch.cat([fts, crop_fts], 1)

        logits = self.logits(fts).view(bs, -1, 3)

        return logits, torch.zeros(bs)


class RNNAttModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        lstm_dim=64,
        dense_dim=64,
        p=0.1,
        num_classes=8,
        num_classes_aux=0,
        n_fts=0,
    ):
        """
        Constructor.

        Args:
            ft_dim (int): The dimension of input features. Defaults to 64.
            lstm_dim (int): The dimension of the LSTM layer. Defaults to 64.
            n_lstm (int): The number of LSTM layers. Defaults to 1.
            dense_dim (int): The dimension of the dense layer. Defaults to 64.
            p (float): Dropout probability. Defaults to 0.1.
            num_classes (int): The number of primary target classes. Defaults to 8.
            num_classes_aux (int): The number of auxiliary target classes. Defaults to 0.
            n_fts (int): The number of additional features. Defaults to 0.

        """
        super().__init__()
        self.n_fts = n_fts
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlps = nn.ModuleDict({
            "nfn": nn.Sequential(nn.Linear(15, dense_dim), nn.Mish()),
            "scs": nn.Sequential(nn.Linear(15, dense_dim), nn.Mish()),
            "ss": nn.Sequential(nn.Linear(6, dense_dim), nn.Mish()),
        })

        self.lstms = nn.ModuleDict({
            k: nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
            for k in ['nfn', 'scs', "ss"]
        })

        if n_fts > 0:  # TODO - non-rnn fts
            self.mlp_fts = nn.Sequential(
                nn.Linear(n_fts, dense_dim),
                nn.Dropout(p=p),
                nn.Mish(),
            )

        # self.dense_all = nn.Sequential(
        #     nn.Dropout(p=0),
        #     nn.Linear(12 * (lstm_dim + dense_dim) + 45, dense_dim),
        #     nn.Mish(),
        # )

        self.logits_scs = nn.Sequential(
            nn.Dropout(p=0),
            # nn.Linear(2 * (2 * lstm_dim + dense_dim) + 15, dense_dim),
            nn.Linear(2 * lstm_dim * 20 + 15, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, dense_dim),
        )

        self.logits_nfn = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * lstm_dim * 20 + 30, dense_dim),
            # nn.Linear(4 * (2 * lstm_dim + dense_dim) + 30, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, dense_dim),
        )

        self.logits_ss = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * lstm_dim * 50, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, dense_dim),
        )

        self.logits = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(dense_dim * 3, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, num_classes),
        )

        if num_classes_aux:
            raise NotImplementedError

    def attention_pooling(self, x, w):
        """
        Apply attention pooling to input features.

        Args:
            x (torch.Tensor): Input feature tensor.
            w (torch.Tensor): Attention weights.

        Returns:
            torch.Tensor: The pooled result.
        """
        return (x * w).sum(1) / (w.sum(1) + 1e-6), (x * w).amax(1)

    def forward(self, x, ft=None):
        """
        Forward pass of the RNN with attention model.
        """
        bs = x['ss'].size(0)

        fts = {}
        for k in self.mlps:
            features = self.mlps[k](x[k])
            features_lstm, _ = self.lstms[k](features)

            # if k == "scs":
            #     fts[k] = torch.cat([features, features_lstm], -1)
            # elif k == "nfn":
            #     fts[k] = torch.cat([features, features_lstm], -1)
            #     fts[k + "_right"] = fts[k][:, :fts[k].size(1) // 2]
            #     fts[k + "_left"] = fts[k][:, fts[k].size(1) // 2:]
            # elif k == "ss":
            fts[k] = features_lstm.reshape(bs, -1)

        scs_fts = torch.cat([fts['scs'], x["scs_crop"]], 1)
        nfn_fts = torch.cat([fts['nfn'], x["nfn_crop"]], 1)

        # scs_fts = torch.cat([fts['scs'].mean(1), fts['scs'].amax(1), x["scs_crop"]], 1)
        # nfn_fts = torch.cat([
        #     fts['nfn_left'].mean(1), fts['nfn_left'].amax(1),
        #     fts['nfn_right'].mean(1), fts['nfn_right'].amax(1),
        #     x["nfn_crop"],
        # ], 1)

        # print(scs_fts.size())
        # print(nfn_fts.size())
        # print(fts['ss'].size())

        logits_scs = self.logits_scs(scs_fts)  # .view(bs, -1, 3)
        logits_nfn = self.logits_nfn(nfn_fts)  # .view(bs, -1, 3)
        logits_ss = self.logits_ss(fts['ss'])  # .view(bs, -1, 3)

        logits = torch.cat([logits_scs, logits_nfn, logits_ss], 1)
        logits = self.logits(logits).view(bs, -1, 3)

        return logits, torch.zeros(bs)
