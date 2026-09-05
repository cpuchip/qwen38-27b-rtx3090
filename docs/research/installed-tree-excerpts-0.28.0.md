# Installed-tree excerpts from the built image qwen38-27b-rtx3090:pr43-6869c80 (vLLM 0.28.0 + fork patches)

Generated 2026-09-05 by grepping the image, so that the installed-tree line numbers cited in docs/v0.28-validation.md can be traced by a reader who has the image and not this box. Each block is the grep and its raw output; empty output means the pattern is absent from that file.

## vllm version: 0.28.0
## config/speculative.py: rejection_sample_method
219:    rejection_sample_method: RejectionSampleMethod = "standard"
232:    non-increasing. Only valid when rejection_sample_method is 'synthetic'.
238:    synthetic_acceptance_rates. Only valid when rejection_sample_method is 'synthetic'.
265:                "rejection_sample_method='synthetic' requires exactly one of "
1365:        if self.rejection_sample_method == "synthetic":
1379:                "are only valid with rejection_sample_method='synthetic'."
## v1/sample/rejection_sampler.py: block
80:            and spec_config.rejection_sample_method == "synthetic"
## #54282 salt (draft gumbel noise decoupling): grep for salt/offset in the draft sampling path
(end salt grep; empty means absent)
## #54374 AOT schedule: sliding-window drafter guard
374:        aot_schedule: bool,
383:        if not aot_schedule:
445:        self.aot_schedule = get_flash_attn_version() == 3
474:        if self.use_full_cuda_graph and self.aot_schedule:
504:        # Sliding window size to be used with the AOT scheduler will be
## async scheduling default
148:    async_scheduling: bool | None = None
172:            if self.async_scheduling:
221:    @field_validator("scheduler_cls", "async_scheduling", mode="wrap")
## MTP proposer chaining (num_speculative_tokens loop)
91:    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
181:    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
184:    Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an
231:    num_speculative_tokens, each entry in [0, 1], and be monotonically
237:    [1, num_speculative_tokens + 1]. Resolved internally to
559:            # same `mtp_num_hidden_layers` field as the multimodal ones.
562:            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
581:            n_predict = getattr(text_config, "mtp_num_hidden_layers", None)
758:        if self.model is None and self.num_speculative_tokens is not None:
797:                    "num_speculative_tokens was provided but without speculative model."
967:                        self.num_speculative_tokens > 1
972:                            "Enabling num_speculative_tokens > 1 will run "

## Addendum 2026-09-05 (Sol second review, section 8 items 2 and 5): the block-verification closure and what mamba_cache_mode align changes

Tree inside the image: /app/venv/lib/python3.12/site-packages/vllm. Each block is the grep and its raw output.

```
$ grep -n "RejectionSampleMethod" config/speculative.py
80:RejectionSampleMethod = Literal["standard", "synthetic", "block"]
219:    rejection_sample_method: RejectionSampleMethod = "standard"

$ sed -n 96,97p v1/worker/gpu/spec_decode/rejection_sampler.py
        elif rejection_sample_method == "block":
            self.use_block_verification = True

$ grep -n "use_block_verification" v1/worker/gpu/spec_decode/rejection_sampler.py
85:        self.use_block_verification: bool = False
97:            self.use_block_verification = True
169:            use_block_verification=self.use_block_verification,

$ grep -n -A6 "mamba_cache_mode" config/cache.py
141:    mamba_cache_mode: MambaCacheMode = "none"
142-    """The cache strategy for Mamba layers:
143-
144-    - "none": set when prefix caching is disabled.
145-    - "all": cache the mamba state of all tokens at position i * block_size.
146-    - "align": only cache the mamba state of the last token of each scheduler step and
147-      when the token is at position i * block_size. This is the default when prefix
--
157:    Requires mamba_cache_mode 'none' or 'align' (prefix caching) and the Triton
158-    mamba backend; standard (non-speculative) decode only. In align mode flushes
159-    are most efficient when mamba_block_size is a multiple of replayssm_buffer_len,
160-    but this is not required."""
161-
162-    # Will be set after profiling.
163-    num_gpu_blocks: int | None = field(default=None, init=False)

$ sed -n 694,712p v1/kv_cache_interface.py

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        # Mamba state is replicated across DCP/PCP ranks, never sharded, so
        # no CP scaling applies.
        if vllm_config.cache_config.mamba_cache_mode == "align":
            # Block table rows are position-indexed over the full sequence
            # even though only 2 + num_speculative_blocks state blocks are
            # resident at a time (earlier states are nulled out by

$ sed -n 1104,1110p v1/worker/gpu_model_runner.py

    def _get_mamba_bufs(self) -> mamba_utils.MambaBuffers:
        # Only reachable on the ``mamba_cache_mode == "align"`` path.
        # The postprocess sub-object is additionally gated on spec
        # decode + hybrid model.
        assert self.cache_config.mamba_cache_mode == "align"
        if self._mamba_bufs is None:

$ sed -n 1635,1642p v1/worker/gpu_model_runner.py
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)

        if self.cache_config.mamba_cache_mode == "align":
            # Fused GPU postprocess: state copies + per-request accepted-token
            # update without CPU-GPU sync. The metadata
            # (num_scheduled_tokens, num_draft_tokens, num_computed_tokens) is
            # pre-staged to GPU buffers in _prepare_inputs.
            mamba_utils.postprocess_mamba_align_gpu(

$ sed -n 2163,2170p v1/worker/gpu_model_runner.py
        # _update_states_after_model_execute for hybrid models).
        # Skipped under async scheduling (non-align): the CPU copy races with
        # the in-flight D2H copy and with input-batch row moves.
        needs_cpu_accepted_counts = self.num_accepted_tokens_event is not None and not (
            self.use_async_scheduling and self.cache_config.mamba_cache_mode != "align"
        )
        if needs_cpu_accepted_counts:
            assert self.num_accepted_tokens_event is not None

$ sed -n 1036,1044p v1/worker/mamba_utils.py
class MambaBuffers:
    """Single owner for all mamba-specific runner buffers.

    The two sub-objects have different gates:
    ``preprocess`` is needed whenever ``mamba_cache_mode == "align"``;
    ``postprocess_align`` is needed only when align is combined with
    speculative decoding on a hybrid model, and is ``None`` otherwise.
    """

```
