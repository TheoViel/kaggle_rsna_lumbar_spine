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
        decoder_channels=(256, 128, 64),  # , 32, 16),
        classes=num_classes,
        use_pixel_shuffle=False,
        use_hypercolumns=False,
        center=False,
        aux_params=None,
    )
    decoder.num_classes = num_classes

    model = SegWrapper(
        decoder,
        use_lstm="lstm" in head_3d,
        two_layers=head_3d.endswith('x2'),
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
        use_lstm=False,
        two_layers=False,
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
        self.num_classes = model.num_classes
        self.num_classes_aux = 0
        self.use_lstm = use_lstm
        self.two_layers = two_layers

        if self.use_lstm:
            self.lstm = nn.LSTM(
                model.encoder.out_channels[-1],
                model.encoder.out_channels[-1] // 2,
                batch_first=True,
                bidirectional=True
            )
            if self.two_layers:
                self.lstm_2 = nn.LSTM(
                    model.encoder.out_channels[-2],
                    model.encoder.out_channels[-2] // 2,
                    batch_first=True,
                    bidirectional=True
                )

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

        if self.use_lstm:
            assert n_frames > 1, "Only one frame, cannot use LSTM / CNN"
            features_ = []
            # frame_idx = self.frames.index(4)
            frame_idx = n_frames // 2

            for i, ft in enumerate(features):
                # print(ft.size())

                if i != len(features) - 1:  # not last layer
                    if self.two_layers and (i == len(features) - 2):
                        pass
                    else:
                        ft = ft.view(bs, n_frames, ft.size(1), ft.size(2), ft.size(3))[:, frame_idx]
                        features_.append(ft)
                        continue

                _, n_fts, h, w = ft.size()
                ft = ft.view(bs, n_frames, n_fts, h, w)

                ft = ft.permute(0, 3, 4, 2, 1).contiguous()  # bs x h x w x n_frames x n_fts
                ft = ft.view(bs * h * w, n_frames, n_fts)

                if i == len(features) - 2:
                    ft = self.lstm_2(ft)[0][:, frame_idx]  # bs x h x w x n_fts
                else:
                    ft = self.lstm(ft)[0][:, frame_idx]  # bs x h x w x n_fts

                ft = ft.view(bs, h, w, n_fts).permute(0, 3, 1, 2)  # bs x n_fts x h x w

                features_.append(ft.view(bs, n_fts, h, w))

            features = features_

        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)

        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
        else:
            labels = torch.zeros(bs, 1).to(x.device)

        return masks, labels
