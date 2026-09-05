From the second measurement campaign (record linked above):

1. The title's hypothesis did not survive: the Triton autotune race does not decide a boot's trajectory. Removing candidate sources made boots diverge more, and boots that loaded identical compiled kernels still diverged.
2. Instead: boot-to-boot divergence exists on both boxes under sampling, never reaches greedy text, is absent at the shipped default with prefix caching on, and grows with prefix caching off or the kernel choice fixed. Its source is unidentified.
3. A bench row needs, beside its cache state: the prefix-caching flag and resolved mamba cache mode, draft width, card, temperature, which graphs the boot loaded or compiled, and how many boots it summarises.

A trajectory is a boot's per-row record of draft rounds, accepted tokens and output length on fixed prompts and seeds, shared when every row matches. Counts are rows of 32 (8 prompts, 4 seeds) on the 4090 host (two RTX 4090s under WSL2), rows of 8 on the native 3090, one boot per arm, sampled unless marked greedy: counts, not statistics.

**Fixing the kernel choice increased divergence.** With VLLM_TRITON_FORCE_FIRST_CONFIG=1, two boots at width 15 on card 1 of the 4090 host matched on 6 of 32 rows against 28 to 31 for untouched pairs on that card; two at width 7 on card 0 were 26 of 30 rows apart, against 0 of 32 autotuned. Registered prediction: 32 of 32. The flag runs at 45.0 and 46.9 ms per step against 23.3 autotuned on card 0, and in both boots prompt 5 returned one token at seeds 2 and 3, as on the native 3090; at temperature 0 it runs to a full answer on both cards: a caution, not a defect.

**Turning prefix caching off increased divergence.** At width 15 on card 1, two cache-off boots matched on 17 of 32 rows against 29 of 32 with it on (registered: 32). Four cache-off boots of that configuration formed three trajectory groups, and the two that loaded the same three compiled-graph artefacts still parted on 18 of 32 rows. One trap: with prefix caching disabled the engine sets the mamba cache mode to none and prints one warning; its earlier arguments line shows the passed value, so every cache-off boot on either box ran at mode none.

**Where the state shows.** At the launcher default (width 7, prefix caching on, sampled) nine boots on card 0 are one trajectory on the recorded counters (no text kept), and a native pair matched 8 of 8. At width 15 with the cache on, five boots on card 0 form two groups separated by the same ten rows, unexplained; the native 3090 gave one group in eight boots. With it off the native box is mixed: two pairs identical 8 of 8, two split.

**Greedy.** Two greedy boots at width 7 on card 1 produced identical text on all 32 rows, so the state never flips an argmax. Against the native 3090's greedy text every prompt parts at one near-tie between the 19th and 53rd word, and the forced kernel configuration moves greedy text on all eight prompts about as much as the card does.

**With a warm cache the race does not run at the default.** Four default boots logged no autotune line. With the cache disabled, two boots re-raced, chose different winners on four kernels of six, and ran at 28.0 and 33.8 ms per step against 23.2 warm, the winner set the only known difference.

**Withdrawn from this thread.** The title's mechanism. "Container timing noise is what makes the draw vary" and "a quiet bare box stays deterministic": the native box diverges too once prefix caching is off. The winner set as what selects the trajectory: kernel configuration moves it, and so do the card and a boot-level state that is neither the race nor prefix caching. "Recording the winners makes a pin unnecessary": winners are one field of what a row needs.

Rule: on this stack a benchmark row is one boot's draw, so persist the cache, publish the row with its cache state, cache mode, width, card and temperature, compare arms only within a boot, and say how many boots agreed.
