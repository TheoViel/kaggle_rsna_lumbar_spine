import torch
import warnings
import torch.nn as nn

from util.torch import load_model_weights

from model_zoo.models import define_model

warnings.simplefilter(action="ignore", category=UserWarning)


def define_model_bi(
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
        ClsModel: The defined model.
    """
    model_1 = define_model(
        name,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
        pretrained=pretrained,
        reduce_stride=reduce_stride,
        increase_stride=increase_stride,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pooling=pooling,
        head_3d=head_3d,
        n_frames=n_frames,
        verbose=verbose,
    )

    model_2 = define_model(
        name,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
        pretrained=pretrained,
        reduce_stride=reduce_stride,
        increase_stride=increase_stride,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pooling=pooling,
        head_3d=head_3d,
        n_frames=n_frames,
        verbose=verbose,
    )
    model_1.logits = nn.Identity()
    model_1.logits_aux = nn.Identity()
    model_2.logits = nn.Identity()
    model_2.logits_aux = nn.Identity()

    model = BiModel(
        model_1,
        model_2,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
    )

    if pretrained_weights:
        model = load_model_weights(
            model, pretrained_weights, verbose=verbose, strict=False
        )

    return model


class BiModel(nn.Module):
    """
    PyTorch model for image classification.

    Attributes:
        encoder: The feature encoder.
        num_classes (int): The number of primary classes.
        num_classes_aux (int): The number of auxiliary classes.
        n_channels (int): The number of input channels.
        drop_rate (float): Dropout rate.
        head_3d (str): The 3D head type.
        n_frames (int): The number of frames.
    """

    def __init__(
        self,
        model_1,
        model_2,
        num_classes=2,
        num_classes_aux=0,
        use_level_embed=False,
    ):
        """
        Constructor for the classification model.
        """
        super().__init__()

        self.model_1 = model_1
        self.model_2 = model_2
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.use_level_embed = use_level_embed

        self.nb_ft = model_1.nb_ft + model_2.nb_ft + 32 * use_level_embed

        self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        if self.use_level_embed:
            self.level_emb = torch.nn.Embedding(5, 32)

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
        fts_1, _ = self.model_1(x["sag"])
        fts_2, _ = self.model_2(x["ax"])
        fts = torch.cat([fts_1, fts_2], 1)

        if self.use_level_embed:
            embed = torch.zeros_like(fts[:, :self.level_emb.embedding_dim])
            embed[:] = self.level_emb(x["level"])
            fts = torch.cat((fts, embed), -1)

        logits, logits_aux = self.get_logits(fts)

        if return_fts:
            return logits, fts
        return logits, logits_aux
