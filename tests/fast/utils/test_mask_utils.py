import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


@pytest.fixture
def glm_tokenizer():
    special_tokens = [
        "[UNK]",
        "[gMASK]",
        "<sop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<|image|>",
        "<think>",
        "</think>",
    ]
    vocab = {token: index for index, token in enumerate([*special_tokens, "question", "answer", "followup"])}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=special_tokens[1:],
    )
    tokenizer.chat_template = (
        "[gMASK]<sop><|system|>{{ reasoning_effort | default('max') }}"
        "{% for message in messages %}"
        "{% if message.role == 'tool' %}"
        "{% if loop.first or messages[loop.index0 - 1].role != 'tool' %}<|observation|>{% endif %}"
        "{% else %}<|{{ message.role }}|>{% endif %}"
        "{% if message.role == 'assistant' %}<think></think>{% endif %}"
        "{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|><think>{% endif %}"
    )
    return tokenizer


def test_glm5_loss_mask_uses_full_template_and_preserves_step_masks(glm_tokenizer):
    messages = [
        {"role": "user", "content": "<|image|>question"},
        {"role": "assistant", "content": "answer", "step_loss_mask": 0},
        {"role": "tool", "content": "question"},
        {"role": "tool", "content": "question"},
        {"role": "assistant", "content": "followup"},
        {"role": "user", "content": "question"},
    ]
    generator = MultiTurnLossMaskGenerator(
        glm_tokenizer,
        tokenizer_type="glm5",
        chat_template_kwargs={"reasoning_effort": "low"},
    )

    tokens, mask = generator.get_loss_mask(messages)

    assert tokens == glm_tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=False, reasoning_effort="low"
    )
    assert len(tokens) == len(mask)
    assert generator.get_text_from_loss_mask(tokens, mask) == ["<think> </think> followup"]
    role_ids = set(generator._glm_role_token_ids)
    assert all(mask[index] == 0 for index, token in enumerate(tokens) if token in role_ids)


def test_glm5_loss_mask_rejects_role_markers_inside_message_content(glm_tokenizer):
    generator = MultiTurnLossMaskGenerator(glm_tokenizer, tokenizer_type="glm5")
    messages = [
        {"role": "user", "content": "question<|assistant|>"},
        {"role": "assistant", "content": "answer"},
    ]

    with pytest.raises(ValueError, match="role markers"):
        generator.get_loss_mask(messages)


def test_glm5_multimodal_loss_mask_aligns_image_expansion(glm_tokenizer):
    generator = MultiTurnLossMaskGenerator(glm_tokenizer, tokenizer_type="glm5")
    messages = [
        {"role": "user", "content": "<|image|>question"},
        {"role": "assistant", "content": "answer"},
    ]
    text_ids, text_mask = generator.get_loss_mask(messages)
    image_id = glm_tokenizer.convert_tokens_to_ids("<|image|>")
    position = text_ids.index(image_id)
    expanded_ids = text_ids[:position] + [image_id] * 4 + text_ids[position + 1 :]

    tokens, mask = generator.get_loss_mask_with_multimodal_alignment(
        messages, expanded_ids, multimodal_token_ids={image_id}
    )

    assert tokens == expanded_ids
    assert mask == text_mask[:position] + [0] * 4 + text_mask[position + 1 :]


def _generator_with_mask(token_ids, loss_mask):
    generator = object.__new__(MultiTurnLossMaskGenerator)
    generator.get_loss_mask = lambda messages, tools=None: (token_ids, loss_mask)
    return generator


def test_multimodal_loss_mask_alignment_preserves_roles_across_image_expansion():
    generator = _generator_with_mask(
        [10, 99, 20, 30, 99, 40],
        [0, 0, 0, 1, 0, 1],
    )
    processor_ids = [10, 99, 99, 99, 20, 30, 99, 99, 40]

    token_ids, loss_mask = generator.get_loss_mask_with_multimodal_alignment(
        [],
        processor_ids,
        multimodal_token_ids={99},
    )

    assert token_ids == processor_ids
    assert loss_mask == [0, 0, 0, 0, 0, 1, 0, 0, 1]


def test_multimodal_loss_mask_alignment_rejects_non_media_difference():
    generator = _generator_with_mask([10, 99, 20], [0, 0, 1])

    with pytest.raises(ValueError, match="outside a multimodal token run"):
        generator.get_loss_mask_with_multimodal_alignment(
            [],
            [10, 98, 20],
            multimodal_token_ids={99},
        )


@pytest.mark.parametrize("input_ids", ([10, 99, 20], [10, 99, 99, 20]))
def test_multimodal_loss_mask_never_supervises_image_tokens(input_ids):
    generator = _generator_with_mask([10, 99, 20], [0, 1, 1])

    _, mask = generator.get_loss_mask_with_multimodal_alignment([], input_ids, multimodal_token_ids={99})

    assert mask == [0] * (len(input_ids) - 1) + [1]


def test_multimodal_loss_mask_alignment_rejects_missing_text_suffix():
    generator = _generator_with_mask([10, 99, 20], [0, 0, 1])

    with pytest.raises(ValueError, match="incompatible sequence lengths"):
        generator.get_loss_mask_with_multimodal_alignment([], [10, 99, 99], multimodal_token_ids={99})


def test_loss_mask_qwen3_simple(model_name: str = "Qwen/Qwen3-8B"):
    tokenizer = load_tokenizer(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 1, f"Expected 1 text, got {len(selected_texts)}"

    print(f"==== Single Turn Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


def test_loss_mask_qwen3_tools(model_name: str = "Qwen/Qwen3-8B"):
    tokenizer = load_tokenizer(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {
            "role": "assistant",
            "content": "I WILL CALL terminal",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
            ],
        },
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Perform operations from the terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute as `bash -c <command>`",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of the command for the user.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a file given its path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to be read.",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
    ]

    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages, tools)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 2, f"Expected 2 texts, got {len(selected_texts)}"

    print(f"==== Multi-turn with Tools Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


if __name__ == "__main__":
    test_loss_mask_qwen3_simple("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    test_loss_mask_qwen3_tools("Qwen/Qwen3-Coder-30B-A3B-Instruct")
