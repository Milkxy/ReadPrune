"""Uniform NeXT allocation, spatial identity, real CLIP and cached generation."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    import transformers
    NEXT_AVAILABLE = transformers.__version__ == "4.57.6"
except ImportError:
    NEXT_AVAILABLE = False

if NEXT_AVAILABLE:
    from readprune.next import NextReadPrune, pack_selected_crops, uniform_crop_budgets


@unittest.skipUnless(NEXT_AVAILABLE, "Install the next extra")
class NextTests(unittest.TestCase):
    def test_uniform_budgets_and_remainders(self):
        for k, quota in ((160, 32), (320, 64), (640, 128)):
            self.assertEqual(uniform_crop_budgets(k, 5, 576), [quota] * 5)
        self.assertEqual(uniform_crop_budgets(163, 5, 576), [33, 33, 33, 32, 32])
        for k in (0, 4, 2881, True, 160.0):
            with self.assertRaises(ValueError):
                uniform_crop_budgets(k, 5, 576)

    def test_spatial_order_and_original_vectors(self):
        # Independent manually enumerated raster for a 2x2 tile grid.
        features = torch.arange(20).reshape(5, 4, 1).double()
        supports = [[1, 3], [0, 3], [0, 2], [1, 2], [0, 3]]
        packed = pack_selected_crops(features, supports, (2, 2))
        self.assertEqual(packed[:, 0].tolist(), [1, 3, 4, 8, 7, 10, 13, 16, 14, 19])
        self.assertEqual(packed.dtype, features.dtype)
        for invalid in ([2, 2], [3, 1], [-1], [4]):
            with self.assertRaises(ValueError):
                pack_selected_crops(features, [invalid] + supports[1:], (2, 2))
        with self.assertRaises(ValueError):
            pack_selected_crops(features, supports, (1, 2))

    def make_runtime(self, dtype=None):
        from transformers import (CLIPConfig, CLIPModel, CLIPVisionConfig, CLIPTextConfig,
                                  LlamaConfig, LlavaNextConfig, LlavaNextForConditionalGeneration,
                                  LlavaNextProcessor, LlavaNextImageProcessor, PreTrainedTokenizerFast)
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        torch.manual_seed(317)
        vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                  num_attention_heads=2, image_size=8, patch_size=2)
        text = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                           max_position_embeddings=256, pad_token_id=0, eos_token_id=62)
        config = LlavaNextConfig(vision_config=vision, text_config=text, image_token_index=63,
                                image_grid_pinpoints=[[16, 16]], vision_feature_layer=-2,
                                vision_feature_select_strategy="default", attn_implementation="eager")
        model = LlavaNextForConditionalGeneration(config).eval().to(dtype=dtype or torch.float32)
        model.generation_config.pad_token_id = 0
        model.generation_config.eos_token_id = None
        paired_config = CLIPConfig(vision_config=vision.to_dict(), text_config=CLIPTextConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, max_position_embeddings=8, bos_token_id=0, eos_token_id=15,
            pad_token_id=1, projection_dim=8).to_dict(), projection_dim=8)
        paired = CLIPModel(paired_config)
        # Save FP32 native values, then let Readout perform its native-dtype check.
        paired.vision_model.load_state_dict(model.model.vision_tower.vision_model.float().state_dict())
        model.to(dtype=dtype or torch.float32)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        paired.save_pretrained(directory.name, safe_serialization=True)

        class ClipTokenizer:
            bos_token_id, eos_token_id, pad_token_id = 0, 15, 1
            def encode(self, text, **kwargs):
                self.last_text = text
                return [3, 4, 5]

        vocabulary = {"[PAD]": 0, "[UNK]": 1, "ask": 2, "answer": 3, **{f"unused_{i}": i for i in range(4, 63)}, "<image>": 63}
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                            unk_token="[UNK]", additional_special_tokens=["<image>"],
                                            model_input_names=["input_ids", "attention_mask"])
        processor = LlavaNextProcessor(LlavaNextImageProcessor(
            size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8},
            image_grid_pinpoints=[[16, 16]]), tokenizer, patch_size=2,
            num_additional_image_tokens=1, vision_feature_select_strategy="default",
            chat_template="<image> {{ messages[0]['content'][1]['text'] }} answer")
        with mock.patch("transformers.CLIPTokenizer.from_pretrained", return_value=ClipTokenizer()):
            runtime = NextReadPrune(model, processor, directory.name)
        self.addCleanup(runtime.close)
        return runtime

    def image_and_inputs(self, runtime):
        import numpy as np
        from PIL import Image
        image = Image.fromarray(np.random.default_rng(17).integers(0, 256, (11, 17, 3), dtype=np.uint8))
        inputs = runtime.processor(images=image, text="<image> ask answer", return_tensors="pt")
        return image, inputs

    def test_real_readout_compaction_and_cached_generation(self):
        for dtype in (torch.float32, torch.bfloat16):
            runtime = self.make_runtime(dtype)
            image, inputs = self.image_and_inputs(runtime)
            compact, audit = runtime.prepare(inputs, "raw question", 17)
            self.assertEqual(audit["crop_budgets"], [4, 4, 3, 3, 3])
            self.assertEqual(audit["K_actual"], 17)
            self.assertEqual(runtime.reader.tokenizer.last_text, "raw question")
            self.assertEqual(compact["inputs_embeds"].shape[1], 19)
            native = runtime.model.model
            with torch.inference_mode():
                features = native.multi_modal_projector(native.vision_tower(
                    inputs["pixel_values"][0].to(dtype), output_hidden_states=True).hidden_states[-2][:, 1:])
                expected = pack_selected_crops(features, audit["supports"], (2, 2))
                torch.testing.assert_close(compact["inputs_embeds"][0, :17], expected, atol=0, rtol=0)
                prefill = runtime.model(**compact, use_cache=True)
                self.assertEqual(prefill.past_key_values.get_seq_length(), 19)
                output = runtime.model.generate(**compact, do_sample=False, max_new_tokens=3)
                self.assertEqual(output.shape, (1, 3))
            result = runtime.generate(image, "raw question", prompt="ask", k=17, max_new_tokens=3)
            self.assertEqual(len(result["generated_ids"]), 3)
            self.assertEqual(runtime.reader.tokenizer.last_text, "raw question")

    def test_full_path_is_native_generation(self):
        runtime = self.make_runtime()
        image, inputs = self.image_and_inputs(runtime)
        with torch.inference_mode():
            expected = runtime.model.generate(**inputs, do_sample=False, max_new_tokens=3)
        result = runtime.generate(image, "ask", k=None, max_new_tokens=3)
        self.assertEqual(result["generated_ids"], expected[0, inputs["input_ids"].shape[1]:].tolist())

    def test_local_loading_fixed_grid_and_public_entrypoint(self):
        runtime = self.make_runtime()
        image, _ = self.image_and_inputs(runtime)
        with tempfile.TemporaryDirectory() as directory:
            runtime.model.save_pretrained(directory, safe_serialization=True)
            runtime.processor.save_pretrained(directory)
            with mock.patch("transformers.CLIPTokenizer.from_pretrained", return_value=runtime.reader.tokenizer):
                with NextReadPrune.from_pretrained(directory, runtime.reader.clip_path,
                        device="cpu", dtype=torch.float32, fixed_grid_size=16) as loaded:
                    result = loaded.generate(image, "ask", k=15, max_new_tokens=2)
                    self.assertEqual(result["audit"]["crop_budgets"], [3] * 5)
                    self.assertEqual(len(result["generated_ids"]), 2)
                    self.assertEqual(loaded.model.config.image_grid_pinpoints, [[16, 16]])

    def test_rejects_misaligned_inputs_and_unsupported_config(self):
        runtime = self.make_runtime()
        _, inputs = self.image_and_inputs(runtime)
        truncated = dict(inputs, input_ids=inputs["input_ids"][:, 1:], attention_mask=inputs["attention_mask"][:, 1:])
        with self.assertRaisesRegex(ValueError, "complete contiguous"):
            runtime.prepare(truncated, "question", 17)
        with self.assertRaisesRegex(ValueError, "crops disagree"):
            runtime.prepare(dict(inputs, pixel_values=inputs["pixel_values"][:, :-1]), "question", 17)
        runtime.model.config.vision_feature_layer = -1
        with self.assertRaisesRegex(ValueError, "feature_layer"):
            NextReadPrune(runtime.model, runtime.processor)


if __name__ == "__main__":
    unittest.main()
