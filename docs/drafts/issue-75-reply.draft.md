**The number you asked for: eight cold boots, eight distinct winner sets, five distinct trajectories.** So it is a long tail on the kernel side and something much smaller on the outcome side, and the gap between those two is the useful part.

Same image, same arguments, cache wiped before each boot so the race re-runs, cache left enabled so the winners are recorded, then the same twelve-turn conversation:

| boots | distinct autotune winner sets | distinct trajectories |
|---:|---:|---:|
| 8 | 8 | 5 |

Trajectories were 58 calls with nothing capped, 74 with one, 160 with seven, 188 with nine, and 189 with nine four times over.

Six kernels are autotuned on this cell, all in the chunked Gated DeltaNet path. Four of them varied across the eight boots, taking three or four distinct configurations each; two never moved at all. No two boots agreed on the full set.

**What that means for a row in a table.** You cannot reproduce a row by hoping to redraw its kernel set, because in eight attempts we never drew the same one twice. But the outcome space is far smaller than the configuration space, so many different winner sets funnel into the same trajectory, and one of them turned up in half the boots. A row is therefore reproducible only with the cache pinned, and a matching number from a fresh boot is weak evidence that you reproduced anything, since the most common attractor comes up roughly half the time by luck.

It also says the mapping is genuinely many-to-one rather than one kernel deciding. That is what killed my own earlier claim: I had pinned a single kernel between two frozen trees and found it decisive there, and across independent draws its configuration was identical in all eight boots while the trajectory ranged over the whole span. Two of the kernels I had reported as inert are among the four that vary.

**On your three channels: agreed, and the separation is better than what I filed.** Only the first is mine. Your second is the one I would most want in the README rather than in a gotchas file, because a benchmark that reports a KV pool without stating its cache state is reporting two different machines under one heading, and 45,000 tokens of context is larger than most of the effects anyone is trying to measure. Your third is the one that caught us both: my first version comparison put its two arms at different positions in the flow and I had to throw the numbers away.

One practical note from running this. Recording the winners is nearly free: the autotune records already sit under the cache directory with every candidate's measured time, so a bench run can copy them beside its row and a later reader can tell whether two rows are even comparable. That is cheaper than pinning and it makes the pin unnecessary for anything that is only being compared against itself.
