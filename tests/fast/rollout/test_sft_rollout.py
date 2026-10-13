import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformers.models.glm46v.video_processing_glm46v import Glm46VVideoProcessor

from miles.rollout import sft_rollout
from miles.utils.data import Dataset
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample
from miles_plugins.models.glm5_next.processor import Glm5NextImageProcessor, Glm5NextProcessor


class _Processor:
    image_token_id = 99


class _Buffer:
    def __init__(self, sample):
        self.sample = sample

    def get_samples(self, count):
        assert count == 1
        return [[self.sample]]


class _MaskGenerator:
    def __init__(self):
        self.calls = []

    def get_loss_mask_with_multimodal_alignment(self, messages, input_ids, tools=None, multimodal_token_ids=None):
        self.calls.append((messages, input_ids, tools, multimodal_token_ids))
        return input_ids, [0, 0, 0, 1, 1]

    def get_response_lengths(self, masks):
        assert masks == [[0, 0, 0, 1, 1]]
        return [2]


def test_generate_rollout_processes_multimodal_sft_sample(monkeypatch):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": "image.png"}]},
        {"role": "assistant", "content": "answer"},
    ]
    sample = Sample(prompt=messages, multimodal_inputs={"images": ["image"], "videos": None})
    processor = _Processor()
    mask_generator = _MaskGenerator()
    pixel_values = torch.ones(4, 8)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    processor_calls = []

    monkeypatch.setattr(sft_rollout, "TOKENIZER", object())
    monkeypatch.setattr(sft_rollout, "PROCESSOR", processor)
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", mask_generator)
    monkeypatch.setattr(sft_rollout, "SAMPLE_PRINTED", False)
    monkeypatch.setattr(sft_rollout.chat_template_utils, "apply_chat_template", lambda *args, **kwargs: "prompt")

    def fake_call_processor(loaded_processor, prompt, multimodal_inputs):
        processor_calls.append((loaded_processor, prompt, multimodal_inputs))
        return {
            "input_ids": [[10, 99, 99, 20, 30]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "mm_token_type_ids": [[0, 1, 1, 0, 0]],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    monkeypatch.setattr(sft_rollout, "call_processor", fake_call_processor)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        apply_chat_template_kwargs=None,
    )

    result = sft_rollout.generate_rollout(args, 0, _Buffer(sample))
    assert result == [[sample]]
    assert processor_calls == [
        (
            processor,
            "prompt",
            {
                "images": ["image"],
                "videos": None,
                "text_kwargs": {"add_special_tokens": False},
            },
        )
    ]
    assert mask_generator.calls == [(messages, [10, 99, 99, 20, 30], None, {99})]
    assert sample.tokens == [10, 99, 99, 20, 30]
    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]
    assert sample.multimodal_train_inputs.keys() == {"pixel_values", "image_grid_thw"}
    assert sample.multimodal_train_inputs["pixel_values"] is pixel_values
    assert sample.multimodal_train_inputs["image_grid_thw"] is image_grid_thw


def test_generate_rollout_empty_supervision_has_empty_response_mask(monkeypatch):
    sample = Sample(prompt=[{"role": "user", "content": "question"}])
    mask_generator = SimpleNamespace(
        get_loss_mask=lambda *args, **kwargs: ([10, 20, 30], [0, 0, 0]),
        get_response_lengths=lambda masks: [0],
    )
    monkeypatch.setattr(sft_rollout, "TOKENIZER", object())
    monkeypatch.setattr(sft_rollout, "PROCESSOR", object())
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", mask_generator)
    monkeypatch.setattr(sft_rollout, "SAMPLE_PRINTED", True)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)

    sft_rollout.generate_rollout(args, 0, _Buffer(sample))

    assert sample.response_length == 0
    assert sample.loss_mask == []
    assert sample.tokens == [10, 20, 30]


def test_generate_rollout_rejects_preformatted_messages(monkeypatch):
    sample = Sample(prompt="preformatted chat")
    monkeypatch.setattr(sft_rollout, "TOKENIZER", object())
    monkeypatch.setattr(sft_rollout, "PROCESSOR", object())
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", object())
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)

    with pytest.raises(ValueError, match="remove --apply-chat-template"):
        sft_rollout.generate_rollout(args, 0, _Buffer(sample))


def test_generate_rollout_rejects_multimodal_data_without_processor(monkeypatch):
    sample = Sample(prompt=[{"role": "user", "content": "image"}], multimodal_inputs={"images": ["image"]})
    monkeypatch.setattr(sft_rollout, "TOKENIZER", object())
    monkeypatch.setattr(sft_rollout, "PROCESSOR", None)
    monkeypatch.setattr(sft_rollout, "load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", object())
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1, hf_checkpoint="/checkpoint")

    with pytest.raises(ValueError, match="supported processor"):
        sft_rollout.generate_rollout(args, 0, _Buffer(sample))


def _image_processor():
    special_tokens = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<|image|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
    ]
    tokens = ["[UNK]", "ignored", "answer", *special_tokens]
    raw = Tokenizer(WordLevel(vocab={token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", additional_special_tokens=special_tokens
    )
    tokenizer.chat_template = (
        "{% for message in messages %}<|{{ message.role }}|>"
        "{% if message.content is string %}{{ message.content }}"
        "{% else %}{% for item in message.content %}"
        "{% if item.type == 'image' %}<|begin_of_image|><|image|><|end_of_image|>"
        "{% elif item.type == 'text' %}{{ item.text }}{% endif %}"
        "{% endfor %}{% endif %}{% endfor %}"
    )
    return Glm5NextProcessor(
        image_processor=Glm5NextImageProcessor(min_image_tokens=1, max_image_tokens=64),
        tokenizer=tokenizer,
        video_processor=Glm46VVideoProcessor(),
    )


def test_dataset_to_image_sft_rollout_preserves_images_and_turn_masks(monkeypatch, tmp_path):
    images = []
    for index, size in enumerate(((28, 28), (56, 28))):
        path = tmp_path / f"image{index}.png"
        Image.new("RGB", size, (255, 0, 0) if index == 0 else (0, 255, 0)).save(path)
        images.append(str(path))
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "<image>"},
                    {"role": "assistant", "content": "ignored", "step_loss_mask": 0},
                    {"role": "user", "content": "<image>"},
                    {"role": "assistant", "content": "answer"},
                ],
                "images": images,
            }
        )
        + "\n"
    )
    processor = _image_processor()
    tokenizer = processor.tokenizer
    dataset = Dataset(
        str(data_path),
        tokenizer,
        processor,
        max_length=None,
        prompt_key="messages",
        multimodal_keys={"image": "images"},
        apply_chat_template=False,
    )
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    monkeypatch.setattr(sft_rollout, "TOKENIZER", tokenizer)
    monkeypatch.setattr(sft_rollout, "PROCESSOR", processor)
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", mask_generator)
    monkeypatch.setattr(sft_rollout, "SAMPLE_PRINTED", True)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1, apply_chat_template_kwargs=None)
    sample = dataset[0]

    sft_rollout.generate_rollout(args, 0, _Buffer(sample))

    train_inputs = sample.multimodal_train_inputs
    assert train_inputs["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
    assert train_inputs["pixel_values"].shape == (12, 3 * 2 * 14**2)
    assert sample.tokens.count(processor.image_token_id) == 3
    assert sample.loss_mask == [1]
    assert sample.response_length == 1
    token_ids, full_mask = mask_generator.get_loss_mask_with_multimodal_alignment(
        sample.prompt, sample.tokens, multimodal_token_ids={processor.image_token_id}
    )
    assert full_mask[token_ids.index(tokenizer.convert_tokens_to_ids("ignored"))] == 0
    assert mask_generator.get_text_from_loss_mask(token_ids, full_mask) == ["answer"]
