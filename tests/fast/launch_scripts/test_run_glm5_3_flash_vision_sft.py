import json
import shlex

import pytest

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend


@pytest.mark.parametrize(
    ("model_name", "num_nodes", "tp", "pp", "ep"),
    [
        ("GLM-5.3-Flash-4layer", 1, "2", "2", "2"),
        ("GLM-5.3-Flash", 4, "8", "4", "8"),
        ("GLM-5.3-Flash", 8, "8", "4", "16"),
    ],
)
def test_image_sft_launcher_has_training_only_model_contract(monkeypatch, model_name, num_nodes, tp, pp, ep):
    launcher = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash_vision_sft.py")
    calls = []
    monkeypatch.setattr(BaseCommandBackend, "execute_train", lambda *args, **kwargs: calls.append(kwargs))

    launcher.execute(launcher.ScriptArgs(model_name=model_name, num_nodes=num_nodes))

    assert len(calls) == 1
    call = calls[0]
    argv = shlex.split(call["train_args"])
    assert call["train_script"] == "train_async.py"
    assert "--debug-train-only" in argv
    assert "--bf16" in argv
    assert "--apply-chat-template" not in argv
    assert "--colocate" not in argv
    assert argv[argv.index("--loss-mask-type") + 1] == "glm5"
    assert argv[argv.index("--multimodal-keys") + 1] == '{"image":"images"}'
    assert argv[argv.index("--tensor-model-parallel-size") + 1] == tp
    assert argv[argv.index("--pipeline-model-parallel-size") + 1] == pp
    assert argv[argv.index("--expert-model-parallel-size") + 1] == ep
    assert not any(flag.startswith("--sglang-") for flag in argv)
    assert "--no-save-optim" not in argv


def test_image_sft_launcher_uses_configured_checkpoint_and_dataset_paths(monkeypatch, tmp_path):
    launcher = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash_vision_sft.py")
    calls = []
    monkeypatch.setattr(BaseCommandBackend, "execute_train", lambda *args, **kwargs: calls.append(kwargs))
    args = launcher.ScriptArgs(
        hf_checkpoint=str(tmp_path / "HF model"),
        ref_load=str(tmp_path / "converted model"),
        prompt_data=str(tmp_path / "image data.jsonl"),
        output_dir=str(tmp_path / "run output"),
    )

    launcher.execute(args)

    argv = shlex.split(calls[0]["train_args"])
    assert argv[argv.index("--hf-checkpoint") + 1] == args.hf_checkpoint
    assert argv[argv.index("--ref-load") + 1] == args.ref_load
    assert argv[argv.index("--prompt-data") + 1] == args.prompt_data
    assert argv[argv.index("--load") + 1] == f"{args.output_dir}/checkpoints"
    assert argv[argv.index("--save") + 1] == f"{args.output_dir}/checkpoints"


def test_image_sft_launcher_can_record_without_checkpoint_or_gpu(monkeypatch):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash_vision_sft.py")

    launcher.execute(launcher.ScriptArgs())

    commands = [command for command in recording.commands if "ray job submit" in command]
    assert len(commands) == 1
    assert "/train_async.py " in commands[0]


def test_image_sft_launcher_honors_checkpoint_rotary_base(monkeypatch, tmp_path):
    launcher = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash_vision_sft.py")
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"rope_theta": 800000}}))
    calls = []
    monkeypatch.setattr(BaseCommandBackend, "execute_train", lambda *args, **kwargs: calls.append(kwargs))

    launcher.execute(launcher.ScriptArgs(hf_checkpoint=str(tmp_path)))

    argv = shlex.split(calls[0]["train_args"])
    assert argv[argv.index("--rotary-base") + 1] == "800000"


def test_image_sft_launcher_explicit_rotary_base_overrides_metadata(tmp_path):
    launcher = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash_vision_sft.py")
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"rope_theta": 800000}}))

    args = launcher.ScriptArgs(hf_checkpoint=str(tmp_path), rotary_base=10000)

    assert args.rotary_base == 10000


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_nodes": 2},
        {"model_name": "GLM-5.3-Flash", "num_nodes": 1},
        {"rollout_batch_size": 4, "global_batch_size": 8},
        {"num_rollout": 0},
    ),
)
def test_image_sft_launcher_rejects_unsupported_topology_and_batch(kwargs):
    launcher = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash_vision_sft.py")

    with pytest.raises(ValueError):
        args = launcher.ScriptArgs(**kwargs)
        _ = args.parallel_args
