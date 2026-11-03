"""Text generation with vLLM (fast, GPU only) or Hugging Face transformers.

Both backends take chat messages, apply the model's chat template, and
return the completions as plain strings, so the rest of the code doesn't
care which one ran.
"""
from __future__ import annotations


def bf16_supported() -> bool:
    """True on GPUs with native bfloat16 (A100 and newer, compute capability 8.0+).

    torch.cuda.is_bf16_supported() also returns True on a T4, where it is only
    emulated and vLLM refuses to load the model.
    """
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _pick_dtype():
    import torch

    if bf16_supported():
        return "bfloat16"
    if torch.cuda.is_available():
        return "float16"  # e.g. Colab T4
    return "float32"


def _pick_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import torch
        import vllm  # noqa: F401

        return "vllm" if torch.cuda.is_available() else "hf"
    except ImportError:
        return "hf"


class Generator:
    def __init__(
        self,
        model: str,
        backend: str = "auto",
        max_new_tokens: int = 512,
        batch_size: int = 32,
        seed: int = 0,
        max_model_len: int = 2048,  # prompt + completion; LiveCodeBench needs more than MBPP
    ):
        from transformers import AutoTokenizer

        self.model_name = model
        self.backend = _pick_backend(backend)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.seed = seed
        self.dtype = _pick_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")

        if self.backend == "vllm":
            from vllm import LLM

            self.llm = LLM(model=model, dtype=self.dtype, seed=seed, max_model_len=max_model_len, gpu_memory_utilization=0.85)
        else:
            import torch
            from transformers import AutoModelForCausalLM

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, self.dtype)).to(self.device)
            self.model.eval()
        print(f"Loaded {model} with backend={self.backend}, dtype={self.dtype}")

    def chat_prompts(self, messages_list: list[list[dict]]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]

    def generate(
        self,
        messages_list: list[list[dict]],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[list[str]]:
        """n completions per conversation. temperature 0 means greedy (then n must be 1)."""
        if temperature == 0 and n > 1:
            raise ValueError("Greedy decoding gives identical samples; use temperature > 0 for n > 1")
        prompts = self.chat_prompts(messages_list)
        if self.backend == "vllm":
            return self._generate_vllm(prompts, n, temperature, top_p)
        return self._generate_hf(prompts, n, temperature, top_p)

    def _generate_vllm(self, prompts, n, temperature, top_p):
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p if temperature > 0 else 1.0,
            max_tokens=self.max_new_tokens,
            seed=self.seed,
        )
        outputs = self.llm.generate(prompts, params)
        return [[o.text for o in out.outputs] for out in outputs]

    def _generate_hf(self, prompts, n, temperature, top_p):
        import torch

        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = lambda x, **_: x  # noqa: E731

        torch.manual_seed(self.seed)
        # Batch prompts of similar length together, then restore the original order.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        results = [None] * len(prompts)
        sample_kwargs = (
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
            if temperature > 0
            else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        )
        for start in tqdm(range(0, len(order), self.batch_size), desc="generate"):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **sample_kwargs,
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for j, i in enumerate(idx):
                results[i] = texts[j * n : (j + 1) * n]
        return results
