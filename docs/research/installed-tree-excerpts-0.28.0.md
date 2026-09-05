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
