# TODO : DOC

import torch
import torch.nn as nn
from model_zoo.unet import Unet


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
    n_frames=1,
    verbose=1,
):
    """
    Loads a pretrained model & builds the architecture.
    Supports timm models.

    Args:
        name (str): Name of the model architecture.
        num_classes (int, optional): Number of main output classes. Defaults to 2.
        num_classes_aux (int, optional): Number of auxiliary output classes. Defaults to 0.
        n_channels (int, optional): Number of input channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pre-trained weights. Defaults to "".
        pretrained (bool, optional): Whether to use pre-trained weights. Defaults to True.
        reduce_stride (bool, optional): Whether to reduce the model's stride. Defaults to False.
        increase_stride (bool, optional): Whether to increase the model's stride. Defaults to False.
        drop_rate (float, optional): Dropout rate. Defaults to 0.
        drop_path_rate (float, optional): Drop path rate. Defaults to 0.
        head_3d (str, optional): 3D head type. Defaults to "".
        n_frames (int, optional): Number of frames. Defaults to 1.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        Unet: The defined model.
    """
    decoder = Unet(
        encoder_name="tu-" + name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=n_channels,
        decoder_channels=(128, 64),  # , 32, 16),
        classes=num_classes_aux,
        use_pixel_shuffle=False,
        use_hypercolumns=False,
        center=False,
        aux_params=None,
    )
    decoder.num_classes = num_classes_aux

    model = SegWrapper(
        decoder,
        head_3d=head_3d,
        num_classes=num_classes,
    )

    if pretrained_weights:
        if verbose:
            print(f'\n-> Loading weights from "{pretrained_weights}"\n')
        state_dict = torch.load(pretrained_weights)
        del (
            state_dict['model.segmentation_head.0.weight'],
            state_dict['model.segmentation_head.0.bias'],
        )
        model.load_state_dict(state_dict, strict=False)

    return model


class SegWrapper(nn.Module):
    """
    Wrapper module for segmentation models.
    """
    def __init__(
        self,
        model,
        head_3d="avg",
        num_classes=1,
    ):
        """
        Constructor.

        Args:
            model (nn.Module): The segmentation model to wrap.
            frames (int or list/tuple, optional): Frame(s) to use. Defaults to 4.
            use_lstm (bool, optional): Whether to use LSTM layer for 2.5D. Defaults to False.
            bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to False.
            two_layers (bool, optional): Whether to use two temporal layers. Defaults to False.
        """
        super().__init__()

        self.model = model
        self.num_masks = model.num_classes
        self.num_classes = num_classes
        self.head_3d = head_3d
        self.nb_ft = model.decoder_out_channels

        if head_3d == "lstm":
            self.lstm = nn.LSTM(
                self.nb_ft, self.nb_ft // 4, batch_first=True, bidirectional=True
            )

        self.logits = nn.Linear(self.model.decoder_out_channels, num_classes)

    def forward_head_3d(self, x):
        """
        Forward function for the 3D head.

        Args:
            x (torch.Tensor [batch_size x n_frames x num_features]): Input features for the 3D head.

        Returns:
            torch.Tensor: Result of the 3D head.
        """
        if self.head_3d == "lstm":
            x, _ = self.lstm(x)
            mean = x.mean(1)
            max_ = x.amax(1)
            x = torch.cat([mean, max_], -1)

        return x

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        bs = x.size(0)
        if len(x.size()) == 5:
            bs, n_frames, c, h, w = x.size()
            x = x.view(bs * n_frames, c, h, w)
        else:
            assert len(x.size()) == 4, "Length of input size not supported"
            bs, c, h, w = x.size()
            n_frames = 1

        features = self.model.encoder(x)

        dec_fts = self.model.decoder(*features)
        dec_fts = dec_fts.view(bs, n_frames, dec_fts.size(1), dec_fts.size(2), dec_fts.size(3))

        masks = self.model.segmentation_head(dec_fts[:, n_frames // 2])

        pooled_fts = (
            dec_fts.unsqueeze(2).repeat(1, 1, self.num_masks, 1, 1, 1) *
            masks.unsqueeze(1).repeat(1, n_frames, 1, 1, 1).unsqueeze(3)
        )  # bs x n_frames x c x ft x h x w
        pooled_fts = pooled_fts.sum((-1, -2))

        pooled_fts = pooled_fts.transpose(1, 2).flatten(0, 1)

        if self.head_3d == "lstm":
            # print(pooled_fts.size())
            pooled_fts = self.forward_head_3d(pooled_fts)
            # print(pooled_fts.size())
        elif self.head_3d == "avg":
            pooled_fts = pooled_fts.mean(1)

        pooled_fts = pooled_fts.view(bs, self.num_masks, -1)

        logits = self.logits(pooled_fts).flatten(1, 2)

        return logits, masks
