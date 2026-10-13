import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.models.glm46v.video_processing_glm46v import Glm46VVideoProcessor

from miles.utils.processing_utils import call_processor
from miles_plugins.models.glm5_next.processor import Glm5NextImageProcessor, Glm5NextProcessor, smart_resize


@pytest.fixture
def processor():
    tokens = [
        "[UNK]",
        "<|image|>",
        "<|video|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab={token: i for i, token in enumerate(tokens)}, unk_token="[UNK]")),
        unk_token="[UNK]",
        additional_special_tokens=tokens[1:],
    )
    return Glm5NextProcessor(
        image_processor=Glm5NextImageProcessor(min_image_tokens=1, max_image_tokens=64),
        tokenizer=tokenizer,
        video_processor=Glm46VVideoProcessor(),
    )


def test_processor_expands_image_tokens_and_preserves_image_order(processor):
    images = [Image.new("RGB", (28, 28), (255, 0, 0)), Image.new("RGB", (56, 28), (0, 255, 0))]
    output = call_processor(
        processor,
        "<|begin_of_image|><|image|><|end_of_image|><|begin_of_image|><|image|><|end_of_image|>",
        {"images": images, "text_kwargs": {"add_special_tokens": False}},
    )

    assert output["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
    assert output["pixel_values"].shape == (12, 3 * 2 * 14**2)
    assert output["input_ids"] == [[3, 1, 4, 3, 1, 1, 4]]
    assert output["mm_token_type_ids"] == [[0, 1, 0, 0, 1, 1, 0]]
    assert torch.isfinite(output["pixel_values"]).all()


def test_processor_extract_media_keeps_original_image_dimensions(processor, tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (101, 47)).save(path)

    media = processor.extract_media([{"role": "user", "content": [{"type": "image", "image": str(path)}]}])

    assert len(media["images"]) == 1
    assert media["images"][0].size == (101, 47)


@pytest.mark.parametrize("modality", ("video", "video_url", "audio", "audio_url", "input_audio"))
def test_processor_extract_media_rejects_non_image_items(processor, modality):
    with pytest.raises(ValueError, match="does not support"):
        processor.extract_media([{"role": "user", "content": [{"type": modality, modality: "source"}]}])


@pytest.mark.parametrize("inputs", ({"videos": ["video"]}, {"audio": [1.0]}, {"audios": [1.0]}))
def test_processor_rejects_video_and_audio(processor, inputs):
    with pytest.raises(ValueError, match="does not support"):
        call_processor(processor, "text", inputs)


@pytest.mark.parametrize("text", ("<|video|>", "<|begin_of_video|>", "<|begin_of_audio|>", "<|end_of_audio|>"))
def test_processor_rejects_video_and_audio_placeholders(processor, text):
    with pytest.raises(ValueError, match="placeholders"):
        call_processor(processor, text)


@pytest.mark.parametrize(
    "content",
    (
        "<|video|>",
        [{"type": "text", "text": "<|begin_of_audio|>"}],
        ["<|end_of_video|>"],
    ),
)
def test_extract_media_rejects_non_image_placeholders_without_images(processor, content):
    with pytest.raises(ValueError, match="placeholders"):
        processor.extract_media([{"role": "user", "content": content}])


@pytest.mark.parametrize(
    ("text", "images"),
    (
        ("<|image|><|image|>", [Image.new("RGB", (28, 28))]),
        ("text", [Image.new("RGB", (28, 28))]),
        ("<|image|>", None),
    ),
)
def test_processor_rejects_image_placeholder_count_mismatch(processor, text, images):
    with pytest.raises(ValueError, match="placeholder/image count mismatch"):
        call_processor(processor, text, {"images": images})


def test_patchify_repeats_each_image_patch_in_temporal_dimension():
    images = torch.arange(16).reshape(1, 1, 4, 4)

    patches, grid_h, grid_w = Glm5NextImageProcessor.patchify(images, 2, 2, 2)

    assert (grid_h, grid_w) == (2, 2)
    assert patches.tolist() == [
        [
            [0, 1, 4, 5, 0, 1, 4, 5],
            [2, 3, 6, 7, 2, 3, 6, 7],
            [8, 9, 12, 13, 8, 9, 12, 13],
            [10, 11, 14, 15, 10, 11, 14, 15],
        ]
    ]


@pytest.mark.parametrize("height,width", ((20, 100), (4096, 2048), (100, 100), (1, 1)))
def test_smart_resize_stays_aligned_and_within_image_token_budget(height, width):
    resized_h, resized_w = smart_resize(2, height, width, factor=28, min_pixels=1, max_pixels=16)

    assert resized_h % 28 == resized_w % 28 == 0
    assert 28**2 <= resized_h * resized_w <= 16 * 28**2
