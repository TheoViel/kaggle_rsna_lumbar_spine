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
    model_classes = {
        "baseline": BaselineModel,
        "simple": SimpleModel,
    }
    model_class = model_classes[name]

    return model_class(
        ft_dim=ft_dim,
        lstm_dim=layer_dim,
        dense_dim=dense_dim,
        resize=resize,
        p=p,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_fts=n_fts,
    )


class SimpleModel(nn.Module):
    def __init__(self, ft_dim=64, dense_dim=64, p=0.0, n_fts=0, **kwargs):
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
        self.num_classes = 3
        self.num_classes_aux = 0

        self.logits_scs = nn.Sequential(
            nn.Linear(ft_dim[0], dense_dim),
            nn.Dropout(p=p),
            nn.LeakyReLU(0.05),
            nn.Linear(dense_dim, 3),
        )

        self.logits_nfn = nn.Sequential(
            nn.Linear(ft_dim[1], dense_dim),
            nn.Dropout(p=p),
            nn.LeakyReLU(0.05),
            nn.Linear(dense_dim, 3),
        )

        self.logits_ss = nn.Sequential(
            nn.Linear(ft_dim[2], dense_dim),
            nn.Dropout(p=p),
            nn.LeakyReLU(0.05),
            nn.Linear(dense_dim, 3),
        )

    def forward(self, x, ft=None):
        """
        Forward pass of the RNN with attention model.
        """
        ref_k = list(x.keys())[0]
        bs = x[ref_k].size(0)

        # print(fts.size())

        logits = torch.zeros(bs, 25, 3).to(x[ref_k].device)

        # fts = x["dd_v1"].view(bs, 25, 3)
        if "crop_2" in x.keys():
            x["crop"] = (x["crop"] + x["crop_2"]) / 2
            del x["crop_2"]
        # x['dh'][:, -10:] = 0
        if "dh_2" in x.keys():
            x['dh'][:, -10:] = x['dh_2'][:, -10:]
            del x['dh_2']

        fts = torch.cat(
            [
                x[k].view(bs, 25, -1)
                for k in x.keys()
                if not any([s in k for s in ["scs", "nfn", "ss", "spinenet"]])
            ],
            -1,
        )

        # print(fts.size())
        # print(x['spinenet'].size())

        fts_scs = torch.cat(
            [fts[:, :5]] +
            [x[k].view(bs, -1, 3) for k in x.keys() if "scs" in k] +
            ([x['spinenet']] if "spinenet" in x else []),
            -1
        )

        fts_nfn = torch.cat(
            [fts[:, 5:15]] +
            [x[k].view(bs, -1, 3) for k in x.keys() if "nfn" in k] +
            ([x['spinenet'].repeat(1, 2, 1)] if "spinenet" in x else []),
            -1
        )

        fts_ss = torch.cat(
            [fts[:, 15:]] +
            [x[k].view(bs, -1, 3) for k in x.keys() if "ss" in k] +
            ([x['spinenet'].repeat(1, 2, 1)] if "spinenet" in x else []),
            -1
        )

        logits[:, :5] = self.logits_scs(fts_scs)
        logits[:, 5:15] = self.logits_nfn(fts_nfn)
        logits[:, 15:] = self.logits_ss(fts_ss)

        # logits[:, 5: 15] = self.logits_nfn(x["nfn_crop_coords"].view(bs, 10, 3))

        return logits, torch.zeros(bs)


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

        logits_fts = 0  # 36 * 2  # Mean & max pool scs/ss/nfn

        self.lstms, self.mlps = {}, {}
        if lstm_dim:
            self.mlps = nn.ModuleDict(
                {
                    "nfn": nn.Sequential(nn.Linear(15, dense_dim), nn.Mish()),
                    "scs": nn.Sequential(nn.Linear(15, dense_dim), nn.Mish()),
                    "ss": nn.Sequential(nn.Linear(6, dense_dim), nn.Mish()),
                }
            )
            self.lstms = nn.ModuleDict(
                {
                    k: nn.LSTM(
                        dense_dim, lstm_dim, batch_first=True, bidirectional=True
                    )
                    for k in ["nfn", "scs", "ss"]
                }
            )
            logits_fts += 4 * lstm_dim * 2 * resize

        self.crop_mlp = None
        if n_fts:
            logits_fts += dense_dim  # * 4
            self.crop_mlp = nn.Sequential(
                nn.Linear(n_fts, dense_dim),
                nn.Dropout(p=p),
                nn.Mish(),
                # nn.Linear(dense_dim, dense_dim),
                # nn.Dropout(p=p),
                # nn.Mish(),
            )

        self.logits = nn.Sequential(
            nn.Linear(logits_fts, dense_dim),
            nn.Dropout(p=p),
            nn.Mish(),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(self, x, ft=None):
        """
        Forward pass of the RNN with attention model.
        """
        ref_k = list(x.keys())[0]
        bs = x[ref_k].size(0)
        fts = torch.empty((bs, 0)).to(x[ref_k].device)

        # print(fts.size())

        fts_lstm = {}
        if len(self.mlps):
            for k in self.mlps:
                features = self.mlps[k](x[k])
                features_lstm, _ = self.lstms[k](features)
                fts_lstm[k] = features_lstm.reshape(bs, -1)

            fts_lstm = torch.cat([fts_lstm[k] for k in fts_lstm], -1)
            fts = torch.cat([fts, fts_lstm], 1)

        # print(fts.size())

        if self.crop_mlp is not None:
            crop_fts = torch.cat([x[k] for k in x if "crop" in k or "dd" in k], 1)
            crop_fts = self.crop_mlp(crop_fts)
            fts = torch.cat([fts, crop_fts], 1)

        # print(fts.size())

        logits = self.logits(fts).view(bs, -1, 3)
        return logits, torch.zeros(bs)
