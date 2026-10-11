from types import SimpleNamespace

from tests.fast.import_isolation_utils import modules_imported_by

from miles_plugins.models.glm5_next.arguments import (
    VISION_TRAINABLE_CHOICES,
    add_glm5_next_vision_arguments,
    vision_scope_may_register_tower,
)

_TOWER_MODULE = "transformers.models.glm_ocr.modeling_glm_ocr"


class TestImageSFTArgumentsStayCheap:
    def test_registering_the_arguments_does_not_pull_in_the_visual_tower(self):
        """miles.utils.arguments builds the parser for every Megatron run, GLM-5.3 or not.

        Importing the tower from here would make every model in the repo depend on the
        Transformers GLM-OCR modeling code, and pay ~1s of import for it.
        """
        modules = modules_imported_by("miles_plugins.models.glm5_next.arguments")

        assert _TOWER_MODULE not in modules
        assert "miles_plugins.models.glm5_next.vision" not in modules
        assert "torch" not in modules

    def test_the_parser_still_offers_every_scope(self):
        """The cheap module owns the scope names, so it has to stay in step with the tower."""
        import argparse

        parser = add_glm5_next_vision_arguments(argparse.ArgumentParser())

        for scope in VISION_TRAINABLE_CHOICES:
            assert parser.parse_args(["--glm5-vision-trainable", scope]).glm5_vision_trainable == scope
        assert parser.parse_args([]).glm5_vision_trainable == "none"
        assert parser.parse_args([]).glm5_vision_recompute is False
        assert parser.parse_args(["--glm5-vision-recompute"]).glm5_vision_recompute is True

    def test_the_tower_and_the_parser_agree_on_the_scope_names(self):
        """Guards the inverted dependency: the two modules must not drift apart."""
        from miles_plugins.models.glm5_next import vision

        assert vision.VISION_TRAINABLE_CHOICES == VISION_TRAINABLE_CHOICES


class TestVisualLoadContextGate:
    def test_only_a_tower_registering_scope_opens_the_gate(self):
        assert vision_scope_may_register_tower(SimpleNamespace(glm5_vision_trainable="projector")) is True
        assert vision_scope_may_register_tower(SimpleNamespace(glm5_vision_trainable="full")) is True

    def test_text_only_runs_keep_it_shut(self):
        """`none` leaves the tower outside the module tree, and other models never set the flag."""
        assert vision_scope_may_register_tower(SimpleNamespace(glm5_vision_trainable="none")) is False
        assert vision_scope_may_register_tower(SimpleNamespace()) is False
