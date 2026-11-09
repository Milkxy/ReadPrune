"""Real tiny HF models: native parity, pruning, DeepStack, and cached decoding."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import transformers
    QWEN_AVAILABLE = transformers.__version__ == "4.57.6"
except ImportError:
    QWEN_AVAILABLE = False

if QWEN_AVAILABLE:
    from readprune.qwen import (FAMILIES, token_budget, visual_features, pack_native,
                               check_native_parity, generate_from_inputs)


def tiny_model(family, dtype=None):
    torch.manual_seed(301)
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=128, pad_token_id=0, eos_token_id=63,
                rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                  patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
                  window_size=8, out_hidden_size=32, fullatt_block_indexes=[0, 1])
    if family == "qwen3_vl":
        vision.update(deepstack_visual_indexes=[0, 1], num_position_embeddings=16)
        config_class = transformers.Qwen3VLConfig
    else:
        config_class = transformers.Qwen2_5_VLConfig
    config = config_class(text_config=text, vision_config=vision, image_token_id=10,
                          video_token_id=12, vision_start_token_id=8, vision_end_token_id=9,
                          attn_implementation="sdpa")
    model = getattr(transformers, FAMILIES[family])(config).to(dtype=dtype or torch.float32).eval()
    model.generation_config.eos_token_id = 63
    return model


def inputs_for(model, prefix=(5,)):
    ids = torch.tensor([[*prefix, 8, 10, 10, 10, 10, 9, 11]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
                pixel_values=torch.randn(16, 24, dtype=model.dtype))


class FixedSupportSelector:
    """Non-prefix support makes a wrong DeepStack or position slice observable."""
    def prepare_image(self, image, grid_hw):
        return grid_hw

    def select(self, features, question, prepared, k):
        import numpy as np
        assert k == 2
        indices = np.array([1, 3])
        return dict(indices=indices, tokens=features[[1, 3]], diagnostics={"raw_question": question})


@unittest.skipUnless(QWEN_AVAILABLE, "Install the qwen extra in its own environment")
class QwenTests(unittest.TestCase):
    def test_explicit_floor_ceil_and_fixed_budget(self):
        self.assertEqual(token_budget(17, retain_ratio=.1, rounding="floor"), 1)
        self.assertEqual(token_budget(17, retain_ratio=.1, rounding="ceil"), 2)
        self.assertEqual(token_budget(17, k=7), 7)
        for kwargs in ({}, {"k": 2, "retain_ratio": .2}, {"k": True}, {"k": 18},
                       {"retain_ratio": 0}, {"retain_ratio": float("nan")},
                       {"retain_ratio": True}, {"k": 1, "rounding": "round"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                token_budget(17, **kwargs)

    def test_real_hf_full_budget_logits_and_generation_parity(self):
        for family in FAMILIES:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(family=family, dtype=dtype):
                    model = tiny_model(family, dtype)
                    for prefix in ((5,), (5, 6, 7)):
                        receipt = check_native_parity(model, inputs_for(model, prefix), max_new_tokens=4)
                        self.assertTrue(receipt["passed"])
                        self.assertIsNone(model.model.rope_deltas)

    def test_nonprefix_selection_preserves_positions_vectors_and_deepstack(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                model = tiny_model(family)
                inputs = inputs_for(model)
                main, deep, _ = visual_features(model, inputs)
                positions, delta = model.model.get_rope_index(
                    inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"],
                    attention_mask=inputs["attention_mask"])
                packed, actual_delta, audit = pack_native(model, inputs, main, deep, [1, 3])
                # Original sequence indices: text=0, start=1, image=2..5, end=6, text=7.
                keep = torch.tensor([0, 1, 3, 5, 6, 7])
                torch.testing.assert_close(packed["position_ids"], positions[:, :, keep], atol=0, rtol=0)
                torch.testing.assert_close(packed["inputs_embeds"][0, [2, 3]], main[[1, 3]], atol=0, rtol=0)
                torch.testing.assert_close(actual_delta, delta + 2, atol=0, rtol=0)
                self.assertEqual(audit["pruned_sequence_length"], 6)
                for original, selected in zip(deep, packed.get("deepstack_visual_embeds", [])):
                    torch.testing.assert_close(selected, original[[1, 3]], atol=0, rtol=0)
                if family == "qwen3_vl":
                    self.assertEqual(audit["deepstack_lengths"], [2, 2])
                    self.assertEqual(packed["visual_pos_masks"].nonzero()[:, 1].tolist(), [2, 3])

    def test_pruned_generation_cache_positions_and_repeated_requests(self):
        for family in FAMILIES:
            model = tiny_model(family, torch.bfloat16)
            for prefix in ((5,), (5, 6, 7)):
                with self.subTest(family=family, prefix=prefix):
                    inputs = inputs_for(model, prefix)
                    kwargs = dict(selector=FixedSupportSelector(), image=object(), raw_question="What?",
                                  k=2, min_new_tokens=4, max_new_tokens=4)
                    result = generate_from_inputs(model, inputs, **kwargs)
                    repeated = generate_from_inputs(model, inputs, **kwargs)
                    self.assertEqual(result["token_ids"], repeated["token_ids"])
                    self.assertEqual(len(result["token_ids"]), 4)
                    audit = result["audit"]
                    self.assertEqual(audit["indices"], [1, 3])
                    self.assertEqual(audit["pruned_sequence_length"], inputs["input_ids"].shape[1] - 2)
                    self.assertEqual(len(audit["decode_checks"]), 3)
                    for check in audit["decode_checks"]:
                        self.assertEqual(check["cache_length"], check["cache_position"] + 1)
                        self.assertEqual(check["position_ids"], [[[check["cache_position"] + audit["rope_delta"]]]] * 3)

    def test_invalid_layout_support_and_missing_selector_fail(self):
        model = tiny_model("qwen3_vl")
        inputs = inputs_for(model)
        main, deep, _ = visual_features(model, inputs)
        for support in ([3, 1], [1, 1], [], [0, 4], [True, 3], [1., 3.]):
            with self.subTest(support=support), self.assertRaises(ValueError):
                pack_native(model, inputs, main, deep, support)
        with self.assertRaises(ValueError):
            pack_native(model, inputs, main, deep[:-1], [1, 3])
        with self.assertRaises(ValueError):
            generate_from_inputs(model, inputs, k=2)
        padded = {**inputs, "attention_mask": inputs["attention_mask"].clone()}
        padded["attention_mask"][0, 0] = 0
        with self.assertRaises(ValueError):
            visual_features(model, padded)
        with self.assertRaises(ValueError):
            visual_features(model, {**inputs, "video_grid_thw": torch.tensor([[1, 4, 4]])})

    def test_local_checkpoint_processor_and_public_entrypoint(self):
        from PIL import Image
        from tokenizers import Tokenizer, models, pre_tokenizers
        from readprune.qwen import QwenReadPrune

        vocab = {f"word{i}": i for i in range(64)}
        for token, index in {"<pad>": 0, "<unk>": 1, "<|vision_start|>": 8,
                             "<|vision_end|>": 9, "<|image_pad|>": 10,
                             "<|video_pad|>": 12, "<|im_end|>": 63}.items():
            del vocab[f"word{index}"]
            vocab[token] = index
        backend = Tokenizer(models.BPE(vocab, merges=[], unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = transformers.Qwen2TokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
            eos_token="<|im_end|>", additional_special_tokens=[
                "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"],
        )
        template = "<|vision_start|><|image_pad|><|vision_end|> {{ messages[0]['content'][1]['text'] }}"
        for family in FAMILIES:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                model = tiny_model(family)
                model.save_pretrained(directory, safe_serialization=True)
                image_processor = transformers.Qwen2VLImageProcessor(
                    patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=16, max_pixels=64)
                if family == "qwen3_vl":
                    processor_class, video_class = transformers.Qwen3VLProcessor, transformers.Qwen3VLVideoProcessor
                else:
                    processor_class, video_class = transformers.Qwen2_5_VLProcessor, transformers.Qwen2VLVideoProcessor
                processor = processor_class(image_processor=image_processor, tokenizer=tokenizer,
                                             video_processor=video_class(), chat_template=template)
                processor.save_pretrained(directory)
                with QwenReadPrune.from_pretrained(directory, device="cpu", min_pixels=16, max_pixels=64) as runtime:
                    self.assertIsNone(runtime.selector)
                    result = runtime.generate(Image.new("RGB", (8, 8)), "word5", retain_ratio=1.0,
                                               check_parity=True, max_new_tokens=3, min_new_tokens=3)
                    self.assertEqual(len(result["token_ids"]), 3)
                    self.assertTrue(result["native_parity"]["passed"])
                    self.assertFalse(result["audit"]["auxiliary_clip_executed"])
                    self.assertIsInstance(result["answer"], str)

    def test_real_auxiliary_clip_to_qwen_generation(self):
        from PIL import Image
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTokenizer
        from transformers.models.clip.tokenization_clip import bytes_to_unicode
        from readprune.qwen_selector import QwenSelector

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            alphabet = list(bytes_to_unicode().values())
            tokens = alphabet + [x + "</w>" for x in alphabet] + ["<|startoftext|>", "<|endoftext|>"]
            (path / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
            (path / "merges.txt").write_text("#version: 0.2\n")
            tokenizer = CLIPTokenizer(vocab_file=str(path / "vocab.json"), merges_file=str(path / "merges.txt"), model_max_length=77)
            tokenizer.save_pretrained(path)
            config = CLIPConfig(projection_dim=8,
                text_config=dict(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77,
                    projection_dim=8, bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id),
                vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                    num_attention_heads=2, image_size=8, patch_size=2, projection_dim=8))
            CLIPModel(config).save_pretrained(path, safe_serialization=True)
            CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}).save_pretrained(path)
            with QwenSelector(path, "cpu") as selector:
                for family in FAMILIES:
                    model = tiny_model(family)
                    result = generate_from_inputs(model, inputs_for(model), selector=selector,
                        image=Image.new("RGB", (12, 8), (20, 80, 150)), raw_question="What is here?",
                        retain_ratio=.5, max_new_tokens=3, min_new_tokens=3)
                    self.assertEqual(result["audit"]["retained_tokens"], 2)
                    self.assertEqual(len(result["token_ids"]), 3)
                    self.assertTrue(result["audit"]["auxiliary_clip_executed"])
                    self.assertTrue(result["audit"]["original_vectors_preserved"])


if __name__ == "__main__":
    unittest.main()
