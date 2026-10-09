"""Launch arguments for GLM-5.3 image SFT."""

from argparse import ArgumentParser

from miles_plugins.models.glm5_next.vision import VISION_TRAINABLE_CHOICES


def add_glm5_next_vision_arguments(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group(title="glm5-next-vision")
    group.add_argument(
        "--glm5-vision-trainable",
        choices=list(VISION_TRAINABLE_CHOICES),
        default="none",
        help=(
            "Which GLM-5.3 visual parameters image SFT trains. 'none' keeps the tower frozen and "
            "outside the optimizer and checkpoint; 'projector' trains the image-feature projection "
            "(post_layernorm, downsample, merger); 'full' also trains patch_embed and the ViT blocks. "
            "Any scope other than 'none' writes visual.* into the training checkpoint."
        ),
    )
    return parser
