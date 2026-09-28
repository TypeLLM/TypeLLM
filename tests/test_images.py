import base64
import io
import os
import pickle
import tempfile
import unittest
from copy import deepcopy

try:
    from PIL import Image
except ImportError:
    Image = None

from typellm import SGLangClient, SGLangError, TypeLLMClient
from typellm.images import encode_image, encode_images

from tests.test_typellm import FakeSGLang

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

# Multi-character pieces, split the way Qwen-style tokenizers split JSON:
# {"k": null} -> '{"' 'k' '":' ' null' '}'. Everything else is one char per token.
PIECES = {'{"': 950, '":': 951, " null": 819, "null": 2827, " -": 900, ' "': 328, ' ""': 901}


def fake_tokenize(text):
    ids, i = [], 0
    while i < len(text):
        piece = max((p for p in PIECES if text.startswith(p, i)), key=len, default=None)
        ids.append(PIECES[piece] if piece else ord(text[i]))
        i += len(piece) if piece else 1
    return ids


def fake_detokenize(ids):
    names = {v: k for k, v in PIECES.items()}
    return "".join(names.get(i) or chr(i) for i in ids)


def decoded(image):
    return Image.open(io.BytesIO(base64.b64decode(encode_image(image).split(",", 1)[1])))


VISION = "<|vision_start|><|image_pad|><|vision_end|>"


class VisionTokenizer:
    """Renders list content the way Qwen-VL chat templates do."""

    chat_template = "vision"
    eos_token = "<|im_end|>"
    eos_token_id = 1
    bos_token = None

    def __init__(self, renders_images=True):
        self.renders_images = renders_images

    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8")

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        out = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    (VISION if self.renders_images else "") if part["type"] == "image" else part["text"]
                    for part in content
                )
            out += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class FakeServerClient(SGLangClient):
    """A real SGLangClient whose HTTP layer answers like SGLang."""

    def __init__(self, tokenizer=None):
        super().__init__(model="fake-vl")
        self._chat_tokenizer = tokenizer or VisionTokenizer()
        self.generate_payloads = []

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/v1/tokenize":
            return {"tokens": fake_tokenize(payload["prompt"])}
        if path == "/v1/detokenize":
            return {"text": fake_detokenize(payload["tokens"])}
        assert path == "/generate", path
        self.generate_payloads.append(payload)
        texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
        ids = payload.get("token_ids_logprob")
        params = payload.get("sampling_params")
        params = params if isinstance(params, list) else [params or {}] * len(texts)
        if ids is None and any("regex" in p for p in params):
            # Grammar-decoded open fields: numbers answer 0, strings "x".
            return [{"text": " 0}" if p["regex"].startswith((" ?-?", " ?(?:-?")) else ' "x"}',
                     "meta_info": {"finish_reason": {"type": "stop"}}} for p in params]
        if ids is None:
            return [{"meta_info": {"prompt_tokens": 900}} for _ in texts]
        ids = [ids] if isinstance(ids[0], int) else ids
        return [
            {"meta_info": {"output_token_ids_logprobs": [[[0.0 if i == 0 else -5.0, t, chr(t)]
                                                          for i, t in enumerate(row)]]}}
            for row in ids
        ]


class ImageEncodingTests(unittest.TestCase):
    def test_bytes_become_a_typed_data_uri(self):
        uri = encode_image(PNG)
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), PNG)

    def test_local_files_are_read_on_the_client(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(PNG)
        try:
            self.assertEqual(encode_image(handle.name), encode_image(PNG))
        finally:
            os.unlink(handle.name)

    def test_urls_and_data_uris_pass_through(self):
        for value in ["https://example.com/a.png", "data:image/png;base64,AAAA"]:
            self.assertEqual(encode_image(value), value)

    def test_rejects_missing_files_and_a_bare_string(self):
        with self.assertRaisesRegex(ValueError, "not a local file"):
            encode_image("missing.png")
        with self.assertRaisesRegex(ValueError, "must be a list"):
            encode_images("receipt.png")


    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_pil_modes_png_cannot_store_fall_back_to_rgb(self):
        for mode in ["CMYK", "YCbCr", "LAB", "HSV", "RGBX", "I;16L"]:
            with self.subTest(mode=mode):
                self.assertEqual(decoded(Image.new(mode, (2, 2))).mode, "RGB")
        self.assertEqual(decoded(Image.new("PA", (2, 2))).mode, "RGBA")

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_pil_modes_png_can_store_are_unchanged(self):
        for mode in ["1", "L", "LA", "P", "RGB", "RGBA", "I;16"]:
            with self.subTest(mode=mode):
                image = Image.new(mode, (2, 2))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                self.assertEqual(encode_image(image), encode_image(buffer.getvalue()))

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_float_and_corrupt_pil_images_raise(self):
        with self.assertRaisesRegex(ValueError, "mode F"):
            encode_image(Image.new("F", (2, 2)))
        buffer = io.BytesIO()
        Image.effect_noise((64, 64), 64).convert("RGB").save(buffer, format="JPEG")
        truncated = Image.open(io.BytesIO(buffer.getvalue()[: len(buffer.getvalue()) // 2]))
        with self.assertRaises(OSError):
            encode_image(truncated)


class PicklingTests(unittest.TestCase):
    def test_clients_pickle_and_copy_without_sharing_images(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="m")
        for copy in (pickle.loads(pickle.dumps(client)), deepcopy(client)):
            self.assertEqual(copy.sglang.model, "m")
            token = client.sglang._active_images.set(("a",))
            try:
                self.assertEqual(copy.sglang._active_images.get(), ())
            finally:
                client.sglang._active_images.reset(token)
            pickle.dumps(copy)


class PlaceholderTests(unittest.TestCase):
    def test_placeholder_is_derived_from_the_template(self):
        self.assertEqual(FakeServerClient().image_placeholder(), VISION)

    def test_text_only_template_is_rejected_before_any_request(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServerClient(VisionTokenizer(renders_images=False))
        with self.assertRaisesRegex(SGLangError, "does not render image content"):
            client.generate(context="Receipt", images=[PNG],
                            questions={"paid": {"type": "boolean", "instructions": "Paid?"}}).result
        self.assertEqual(client.sglang.generate_payloads, [])

    def test_context_containing_image_tokens_is_rejected(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServerClient()
        with self.assertRaisesRegex(SGLangError, "2 image placeholders for 1 images"):
            client.generate(context=VISION, images=[PNG],
                            questions={"paid": {"type": "boolean", "instructions": "Paid?"}}).result


class ImageRequestTests(unittest.TestCase):
    QUESTIONS = {
        "paid": {"type": "boolean", "instructions": "Was it paid?"},
        "category": {"type": "string", "enum": ["office", "food"], "instructions": "Classify."},
    }

    def run_generate(self, images, **kwargs):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServerClient()
        client.generate(context="Receipt", images=images, **kwargs).result
        return client.sglang.generate_payloads

    def assert_images_attached(self, payloads, images):
        self.assertTrue(payloads)
        for payload in payloads:
            texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
            if isinstance(payload["text"], str):
                expected = list(images)
            elif len(images) == 1:
                expected = images[0]
            else:
                expected = [list(images)] * len(texts)
            self.assertEqual(payload["image_data"], expected)
            for text in texts:
                self.assertEqual(text.count(VISION), len(images))
                # Images come before the context text in the first user turn.
                self.assertLess(text.index(VISION), text.index("Receipt"))

    def test_every_batch_request_carries_the_images(self):
        images = [encode_image(PNG), "https://example.com/b.png"]
        self.assert_images_attached(self.run_generate(images, questions=self.QUESTIONS), images)

    def test_a_single_image_is_sent_once_per_batch(self):
        image = encode_image(PNG)
        questions = {"n": {"type": "integer", "enum": [1, 2, 3, 4], "permutations": "all"}}
        payloads = self.run_generate([image], questions=questions)
        self.assert_images_attached(payloads, [image])
        [scored] = [p for p in payloads if not isinstance(p["text"], str) and len(p["text"]) == 24]
        self.assertEqual(scored["image_data"], image)

    def test_dependency_requests_carry_the_images(self):
        dag = {**self.QUESTIONS, "advice": {"type": "boolean", "instructions": "Refund?",
                                            "depends_on": ["paid", "category"]}}
        images = [encode_image(PNG)]
        self.assert_images_attached(self.run_generate(images, questions=dag), images)

    def test_numeric_decoding_carries_the_images(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServerClient()
        client.generate(context="Receipt", images=[PNG],
                        questions={"n": {"type": "integer", "instructions": "Count?"}}).result
        self.assert_images_attached(client.sglang.generate_payloads, [encode_image(PNG)])

    def test_images_are_scoped_to_one_call(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServerClient()
        client.generate(context="Receipt", images=[PNG], questions=self.QUESTIONS).result
        client.sglang.generate_payloads.clear()
        client.generate(context="Receipt", questions=self.QUESTIONS).result
        self.assertTrue(all("image_data" not in p for p in client.sglang.generate_payloads))

    def test_thinking_budget_counts_image_tokens_on_the_server(self):
        client = FakeServerClient()
        client.thinking_budget = None
        client._context_length_cache = 1000
        prefix = VisionTokenizer().apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Receipt"}]}],
            add_generation_prompt=True) + "<think>\n"
        client._chat_tokenizer.chat_template = "<think></think>"
        with client.images([encode_image(PNG)]):
            with self.assertRaisesRegex(SGLangError, "no room for thinking"):
                client._finish_thinking(prefix)
        [warm, *rest] = client.generate_payloads
        self.assertEqual(warm["sampling_params"]["max_new_tokens"], 0)
        # 900 served prompt tokens leave 1000 - 900 - reserve - close - 16 < 0,
        # so thinking must not be requested at all.
        self.assertEqual(rest, [])


class ImageInputTokenTests(unittest.TestCase):
    def test_images_count_as_the_server_expands_them(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake-vl")
        client.sglang = FakeServerClient()
        questions = {"a": {"type": "boolean"}}
        usage = client.generate(context="Receipt", images=[PNG], questions=questions).usage
        first = client.sglang.generate_payloads[0]["text"]
        first = first if isinstance(first, str) else first[0]
        count = client.sglang.count_tokens
        # The server reported 900 prompt tokens, with the placeholder expanded.
        image = 900 - count(first) + count(VISION)
        self.assertEqual(usage.input_tokens,
                         count("Receipt") + count('{"a": {"type": "boolean"}}') + image)


class RuntimeContentTests(unittest.TestCase):
    def test_fake_backends_receive_image_parts(self):
        class Recording(FakeSGLang):
            def __init__(self):
                super().__init__(selected_ids=[ord("A")] * 4)
                self.messages = []

            def images(self, images):
                from contextlib import nullcontext
                self.attached = images
                return nullcontext()

            def render_chat(self, messages, *, add_generation_prompt):
                self.messages.append(messages)
                return "x"

        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = Recording()
        client.generate(context="Receipt", images=[PNG, PNG],
                        questions={"paid": {"type": "boolean", "instructions": "Paid?"}}).result
        first = client.sglang.messages[0][0]["content"]
        self.assertEqual(first, [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Receipt"}])
        self.assertEqual(client.sglang.attached, (encode_image(PNG),) * 2)


if __name__ == "__main__":
    unittest.main()
