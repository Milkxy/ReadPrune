"""CPU readout tests with tiny random fixtures; no real weights or downloads."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
    from torch import nn
    from safetensors.torch import load_file, save_file
    from transformers import CLIPConfig, CLIPModel, CLIPVisionModel
    from readprune.readout import Readout, ablate_cls_connections, fp32_scope, segment_content
except ImportError:
    torch = None


def tiny_readout():
    torch.manual_seed(19)
    r = Readout.__new__(Readout)
    r.device = torch.device("cpu")
    attn = nn.Module()
    attn.num_heads, attn.head_dim, attn.scale = 2, 4, 4 ** -.5
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        setattr(attn, name, nn.Linear(8, 8))
    r.block = SimpleNamespace(
        layer_norm1=nn.LayerNorm(8), self_attn=attn, layer_norm2=nn.LayerNorm(8),
        mlp=nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)),
    )
    r.post_ln = nn.LayerNorm(8)
    r.visual_projection = nn.Linear(8, 6, bias=False)
    return r


@unittest.skipIf(torch is None, "Install the readout extra to run PyTorch/Transformers tests")
class ReadoutTests(unittest.TestCase):
    def test_torch_selector_gathers_exact_original_rows(self):
        from readprune import select_tokens

        generator = torch.Generator().manual_seed(51)
        features = torch.randn(21, 12, generator=generator).half()
        delta = torch.randn(21, generator=generator)
        before, delta_before = features.clone(), delta.clone()
        result = select_tokens(features, delta, 7)
        indices = torch.as_tensor(result["indices"], dtype=torch.long)
        expected = features.index_select(0, indices)
        torch.testing.assert_close(result["tokens"], expected, atol=0, rtol=0)
        torch.testing.assert_close(features, before, atol=0, rtol=0)
        torch.testing.assert_close(delta, delta_before, atol=0, rtol=0)
        self.assertEqual(result["tokens"].dtype, features.dtype)
        self.assertEqual(result["tokens"].device, features.device)
        self.assertEqual(indices.tolist(), sorted(set(indices.tolist())))
        self.assertEqual(len(indices), 7)

    def test_edge_ablation_matches_explicit_softmax(self):
        torch.manual_seed(17)
        logits = torch.randn(3, 9)
        values = torch.randn(3, 9, 5)
        base, removed, info = ablate_cls_connections(logits, values)
        expected_base = torch.softmax(logits, -1).unsqueeze(1).bmm(values).squeeze(1)
        torch.testing.assert_close(base, expected_base, atol=0, rtol=0)
        for patch in range(8):
            masked = logits.clone()
            masked[:, patch + 1] = -torch.inf
            expected = torch.softmax(masked, -1).unsqueeze(1).bmm(values).squeeze(1)
            torch.testing.assert_close(removed[patch], expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(info["cls_key_included"])

    def test_saturated_attention_uses_exact_masked_fallback(self):
        logits = torch.tensor([[0., 1000., -1., 1.]])
        values = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]])
        _, removed, info = ablate_cls_connections(logits, values)
        masked = logits.clone()
        masked[:, 1] = -torch.inf
        expected = torch.softmax(masked, -1).unsqueeze(1).bmm(values).squeeze(1)
        torch.testing.assert_close(removed[0], expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(info["fallback_head_candidate_pairs"], 1)
        self.assertTrue(torch.isfinite(removed).all())

    def test_batched_nonlinear_readout_matches_explicit_candidates(self):
        r = tiny_readout()
        H = torch.randn(1, 17, 8)
        before = H.clone()
        actual = r.image(H)
        hidden = r.block.layer_norm1(H)
        attn = r.block.self_attn
        def shape(x):
            return x.view(1, 17, 2, 4).transpose(1, 2).contiguous()[0]
        logits = (shape(attn.q_proj(hidden) * attn.scale)[:, :1]
                  @ shape(attn.k_proj(hidden)).transpose(1, 2)).squeeze(1)
        values = shape(attn.v_proj(hidden))
        references = []
        for patch in (None, *range(16)):
            masked = logits.clone()
            if patch is not None:
                masked[:, patch + 1] = -torch.inf
            heads = torch.softmax(masked, -1).unsqueeze(1).bmm(values).squeeze(1)
            references.append(r._complete_cls(H[0, :1], heads.flatten()[None])[0])
        expected = torch.stack(references)
        torch.testing.assert_close(actual["e_full"], expected[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual["R"], expected[:1] - expected[1:], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(H, before, atol=0, rtol=0)

    def test_text_segmentation_does_not_truncate(self):
        content = list(range(20))
        chunks = segment_content(content, 100, 101, 8)
        self.assertEqual([v for chunk in chunks for v in chunk["input_ids"][1:-1]], content)
        self.assertEqual([c["content_tokens"] for c in chunks], [6, 6, 6, 2])
        self.assertTrue(all(c["input_ids"][c["eot_position"]] == 101 for c in chunks))
        self.assertEqual(segment_content([], 100, 101, 8)[0]["input_ids"], [100, 101])

    def test_fp32_scope_restores_global_options_on_failure(self):
        dtype = torch.get_default_dtype()
        matmul = torch.backends.cuda.matmul.allow_tf32
        cudnn = torch.backends.cudnn.allow_tf32
        with self.assertRaisesRegex(ValueError, "fixture"):
            with fp32_scope():
                self.assertEqual(torch.get_default_dtype(), torch.float32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                raise ValueError("fixture")
        self.assertEqual(torch.get_default_dtype(), dtype)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, matmul)
        self.assertEqual(torch.backends.cudnn.allow_tf32, cudnn)


@unittest.skipIf(torch is None, "Install the readout extra to run checkpoint tests")
class SafeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="readprune-test-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        torch.manual_seed(119)
        self.config = CLIPConfig(
            projection_dim=4,
            text_config=dict(vocab_size=16, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2,
                             max_position_embeddings=8, projection_dim=4,
                             bos_token_id=0, eos_token_id=2, pad_token_id=1),
            vision_config=dict(hidden_size=8, intermediate_size=16,
                               num_hidden_layers=1, num_attention_heads=2,
                               image_size=4, patch_size=2, projection_dim=4),
        )
        self.config.save_pretrained(self.path)
        paired = CLIPModel(self.config)
        self.state = {key: value.detach().cpu().contiguous().clone()
                      for key, value in paired.state_dict().items()}
        self.native = CLIPVisionModel(self.config.vision_config).eval()
        self.native.load_state_dict({key: value for key, value in self.state.items()
                                     if key.startswith("vision_model.")}, strict=True)
        self.tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=2)

    def write_safe(self):
        save_file(self.state, str(self.path / "model.safetensors"), metadata={"format": "pt"})

    def construct(self):
        with mock.patch("torch.load", side_effect=AssertionError("pickle loading forbidden")), \
             mock.patch("transformers.CLIPTokenizer.from_pretrained", return_value=self.tokenizer):
            return Readout(self.native, self.path, "cpu")

    def test_loads_safetensors_preserves_pairing_and_native_values(self):
        self.write_safe()
        before = {k: v.clone() for k, v in self.native.state_dict().items()}
        with mock.patch("safetensors.torch.load_file", wraps=load_file) as loader:
            readout = self.construct()
        self.addCleanup(readout.close)
        loader.assert_called_once_with(str(self.path / "model.safetensors"), device="cpu")
        self.assertTrue(readout.metadata["alignment_pass"])
        for key, value in self.native.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        torch.testing.assert_close(readout.visual_projection.weight,
                                   self.state["visual_projection.weight"], atol=0, rtol=0)
        # Actual tiny HF CLIP block checks independent explicit masks against
        # the batched algebra, with no downloaded model or CUDA allocation.
        H = torch.randn(1, 5, 8)
        self.assertTrue(readout.verify(H, patch_ids=(0, 1, 2, 3))["pass"])

    def test_missing_safe_file_never_falls_back_to_pickle(self):
        (self.path / "pytorch_model.bin").write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(RuntimeError, "paired model.safetensors missing"):
            self.construct()

    def test_mismatched_native_checkpoint_is_rejected(self):
        self.state["vision_model.embeddings.class_embedding"][0] += 1
        self.write_safe()
        with self.assertRaisesRegex(RuntimeError, "native differs from checkpoint"):
            self.construct()

    def test_missing_visual_projection_is_rejected(self):
        self.state.pop("visual_projection.weight")
        self.write_safe()
        with self.assertRaisesRegex(RuntimeError, "visual_projection missing"):
            self.construct()

    def test_missing_text_weights_are_rejected(self):
        self.state.pop("text_model.embeddings.token_embedding.weight")
        self.write_safe()
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.construct()

    def test_long_question_uses_explicit_eot_and_content_length_weights(self):
        # Give EOT a distinct highest ID, matching paired CLIP tokenization.
        # Fourteen content tokens form uneven chunks of lengths 6, 6, and 2.
        self.config.text_config.eos_token_id = 15
        self.config.save_pretrained(self.path)

        class TinyTokenizer:
            bos_token_id = 0
            eos_token_id = 15
            pad_token_id = 1

            def encode(self, text, add_special_tokens=False, truncation=False):
                assert add_special_tokens is False and truncation is False
                return [int(value) for value in text.split()]

        self.tokenizer = TinyTokenizer()
        self.write_safe()
        readout = self.construct()
        self.addCleanup(readout.close)
        content = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2]
        question = " ".join(map(str, content))
        actual = readout.text(question)

        # Independently run each legal chunk through the real random CLIP text
        # transformer, selecting its known EOT position rather than relying on
        # the batching or chunking code under test.
        vectors, lengths = [], []
        with torch.inference_mode():
            for start in range(0, len(content), 6):
                piece = content[start:start + 6]
                ids = torch.tensor([[0, *piece, 15]], dtype=torch.long)
                output = readout.text_model.text_model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True)
                pooled = output.last_hidden_state[:, len(piece) + 1]
                torch.testing.assert_close(pooled, output.pooler_output, atol=0, rtol=0)
                projected = readout.text_model.text_projection(pooled)[0]
                vectors.append(projected / projected.norm())
                lengths.append(len(piece))
            vectors = torch.stack(vectors)
            weighted = (vectors * torch.tensor(lengths)[:, None]).sum(0)
            expected = weighted / weighted.norm()
            unweighted = vectors.mean(0)
            unweighted = unweighted / unweighted.norm()

        torch.testing.assert_close(actual["t"], expected, atol=2e-6, rtol=2e-6)
        self.assertGreater(float(torch.linalg.vector_norm(expected - unweighted)), 1e-5)
        self.assertEqual(actual["diag"]["segment_content_lengths"], [6, 6, 2])
        self.assertEqual(actual["diag"]["eot_positions"], [7, 7, 3])
        self.assertFalse(actual["diag"]["truncated"])
        recovered = [v for ids in actual["diag"]["input_ids"] for v in ids[1:-1]]
        self.assertEqual(recovered, content)


if __name__ == "__main__":
    unittest.main()
