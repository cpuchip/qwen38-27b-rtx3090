# Public on-demand GPU rental prices, fetched 2026-09-05

Purpose: costing a drafter-training run for Qwen3.8-27B (DFlash2-style head, five
transformer layers at hidden 5120, about 1.2 GB at 4-bit, against a 27B target).
Rows that matter most: 24 GB, 48 GB, 80 GB cards; on-demand and spot where the
provider publishes spot; multi-GPU instance prices where a page lists them.

All prices are USD per GPU per hour unless a row says otherwise. Every number
below was read from the named URL on 2026-09-05. Nothing is from memory.

## How each provider was verified

| Provider | Primary source | Status on 2026-09-05 |
|---|---|---|
| RunPod | https://www.runpod.io/pricing | Fetched, prices present in page |
| Vast.ai | https://vast.ai/pricing | JS-only, NO prices in static HTML. Fell back to Vast's own live offers API (see section 2), plus dated secondary pages |
| Lambda | https://lambda.ai/service/gpu-cloud and https://lambda.ai/pricing | Fetched, prices present. The old lambdalabs.com/service/gpu-cloud URL returns HTTP 404 |
| Together AI | https://www.together.ai/pricing and https://www.together.ai/gpu-clusters | Fetched, prices present |

Vast.ai caveat: direct curl / urllib from this box to the Vast API returned HTTP 403,
so the API JSON was read through the WebFetch tool's summarizer rather than parsed by
my own script. The numbers are from the live marketplace on the real path, but a
summarizer sat between the JSON and me.

---

## 1. RunPod

URL: https://www.runpod.io/pricing (fetched 2026-09-05)
Billing docs: https://docs.runpod.io/pods/pricing (fetched 2026-09-05)
Pod-type docs: https://docs.runpod.io/pods/choose-a-pod (fetched 2026-09-05)

Table columns on the page are literally "Community Cloud" and "Secure Cloud".
A "Per hour / Per second" toggle sits above the table. Prices are per GPU.

```
GPU                 VRAM    Community $/GPU-hr   Secure $/GPU-hr   Type
RTX 4090            24 GB   0.34                 0.74              on-demand
RTX 5090            32 GB   0.69                 0.99              on-demand
RTX A5000           24 GB   0.16                 0.27              on-demand
A40                 48 GB   0.35                 0.44              on-demand
L40                 48 GB   0.69                 0.82              on-demand
L40S                48 GB   0.79                 0.99              on-demand
RTX 6000 Ada        48 GB   0.74                 0.84              on-demand
RTX A6000           48 GB   0.33                 0.53              on-demand
RTX PRO 6000        96 GB   1.69                 2.09              on-demand
A100 PCIe           80 GB   1.19                 1.39              on-demand
A100 SXM            80 GB   1.39                 1.59              on-demand
A100 40 GB          --      not listed           not listed
H100 PCIe           80 GB   1.99                 2.89              on-demand
H100 SXM            80 GB   2.69                 3.29              on-demand
H100 NVL            94 GB   2.59                 3.19              on-demand
H200                141 GB  3.59                 4.59              on-demand
B200                180 GB  5.98                 6.79              on-demand
B300                288 GB  6.94                 7.89              on-demand
```

RunPod notes:

- Per GPU, not per node. Multi-GPU pods are N times the per-GPU price; no
  multi-GPU discount or premium is shown on the page.
- Billing: docs say "Pods are billed by the second for compute and storage, with
  no fees for data ingress or egress." Account needs at least one hour of credits
  to start a pod. No minimum rental duration.
- Storage (docs and page agree):
  - Container disk: $0.10/GB/month (not charged while pod is stopped)
  - Volume disk: $0.10/GB/month running, $0.20/GB/month stopped
  - Network volume: $0.07/GB/month under 1 TB, $0.05/GB/month over 1 TB
  - High-performance network volume: $0.14/GB/month
- Egress: none.
- Spot: NOT on the pricing page and NOT on the docs pricing page. The only RunPod
  source is a blog post dated August 25, 2026
  (https://www.runpod.io/blog/spot-vs-on-demand-instances-runpod) saying spot is
  "usually much cheaper (50%)" and "can be interrupted without notice". Its worked
  example (A6000 spot $0.232 vs on-demand $0.491) does not match today's page, so
  treat the 50% figure as the only usable part.
- Savings plans: 3-month or 6-month prepaid terms for compute discounts, storage
  excluded, non-refundable. Discount amounts not published.
- Docs say "Runpod is no longer accepting new hosts for Community Cloud. Existing
  Community Cloud resources remain available." Community-tier stock may shrink.

---

## 2. Vast.ai

Pricing page https://vast.ai/pricing: JS-only, no prices in static HTML
(fetched 2026-09-05). Per-GPU pages such as https://vast.ai/pricing/gpu/RTX-4090
and https://vast.ai/pricing/gpu/H100-SXM are also blank in static HTML; the
heading reads "Rent RTX 4090 GPUs on Vast.ai for" followed by nothing.

Primary data instead: Vast's public live offers API
https://console.vast.ai/api/v0/bundles/ queried per GPU with filter
verified hosts, rentable, exactly 1 GPU, sorted by price ascending
(fetched 2026-09-05 about 12:20 UTC). The API returns at most 40 offers per
query, so "median" is the median of the cheapest offers returned, not of the
whole market. Price field is dph_total: dollars per hour for the whole
1-GPU instance including its default disk allocation.

```
GPU                 VRAM     Min $/hr   Median $/hr   Offers   Type
RTX 4090            24 GB    0.223      0.435         40       on-demand
RTX 5090            32 GB    0.343      0.627         40       on-demand
RTX A6000           48 GB    0.402      0.463         4        on-demand
RTX 6000 Ada        48 GB    0.515      0.625         2        on-demand
L40S                48 GB    0.801      0.801         2        on-demand
RTX PRO 6000 WS     96 GB    1.341      1.602         20       on-demand
A100 SXM4 (40/80)   mixed    0.564      0.828         9        on-demand  (min is a 40 GB card)
A100 PCIe (40/80)   mixed    0.668      0.901         3        on-demand  (min is a 40 GB card)
H100 SXM            80 GB    2.406      4.341         4        on-demand
H100 PCIe           80 GB    2.935      3.002         2        on-demand
H100 NVL            94 GB    2.670      2.838         4        on-demand
H200                141 GB   3.977      4.490         2        on-demand
B200                180 GB   6.010      7.752         3        on-demand

RTX 4090            24 GB    0.158      0.296         30       interruptible (bid)
A100 SXM4           40 GB    0.401      0.535         4        interruptible (bid)
H100 SXM            80 GB    0.673      2.371         4        interruptible (bid)
```

Full RTX 4090 on-demand price list returned (verified, 1x, cheapest 40, the
summarizer listed 24 of them): 0.223, 0.311, 0.313, 0.322, 0.336, 0.336, 0.362,
0.375, 0.382, 0.382, 0.391, 0.402, 0.404, 0.404, 0.468, 0.468, 0.492, 0.601,
0.607, 0.607, 0.670, 0.670, 0.681, 0.710.

Full RTX 4090 interruptible list (30 offers): 0.158, 0.158, 0.159, 0.159, 0.201,
0.203, 0.204, 0.204, 0.214, 0.215, 0.215, 0.268, 0.269, 0.269, 0.295, 0.296,
0.332, 0.334, 0.335, 0.350, 0.361, 0.362, 0.403, 0.404, 0.404, 0.468, 0.541,
0.603, 0.789, 0.789.

Vast.ai notes:

- Marketplace: every price is host-set and moves hourly. Verified-host inventory
  for datacenter cards was thin today (2 to 9 offers each), so those medians are
  noisy. Unverified hosts are usually cheaper but were excluded from the query.
- Per GPU instance. Multi-GPU offers exist (2x, 4x, 8x) but were not queried;
  they are separate listings, not a multiple of the 1x price.
- Billing docs https://docs.vast.ai/guides/instances/pricing (fetched 2026-09-05):
  "Billed by the second for actual usage." Credits required upfront. No minimum
  duration stated.
- Storage: charged separately, rate set by each host, "typically higher for
  stopped instances than running instances." Storage keeps billing while an
  instance is stopped; delete the instance to stop it.
- Bandwidth: charged per byte transferred at host-set rates, both directions.
  Docs advise reviewing bandwidth rates at instance selection.
- Interruptible: docs say "often 50%+ cheaper than on-demand." Client sets a bid;
  the highest bid runs, lower bids are paused. Host sets a minimum bid.
- Secondary sources, labeled as secondary:
  - Thunder Compute blog, dated September 1, 2026
    (https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute): Vast RTX 4090
    $0.39, A100 80GB $1.32, H100 $2.21, RTX A6000 $0.41, interruptible "30-50%+
    cheaper".
  - GetDeploying, page says "Sept. 5, 2026" (https://getdeploying.com/gpus/nvidia-rtx-4090
    and https://getdeploying.com/gpus/nvidia-h200): Vast RTX 4090 from $0.28,
    Vast H200 NVL from $3.34.
  - ComputeComparison, undated (https://computecomparison.com/provider/vast-ai):
    RTX 4090 $0.35 on-demand / $0.14 spot, RTX 5090 $1.20 / $0.45, A100 80GB
    $0.95 / $0.38, H100 80GB $2.00 / $0.81, L40S $0.48 / $0.19; states spot is
    "typically 40-70% cheaper than on-demand rates."
  - A search-engine cached title for the Vast RTX 4090 page read "Rent RTX 4090
    GPUs on Vast.ai for $0.12/hr"; that is an indexer snapshot of the live
    minimum at some unknown date, not something I could see on the page.

---

## 3. Lambda

URL: https://lambda.ai/service/gpu-cloud (fetched 2026-09-05)
Cluster and same instance table: https://lambda.ai/pricing (fetched 2026-09-05)
Billing docs: https://docs.lambda.ai/public-cloud/billing/ (fetched 2026-09-05)
Instance-type docs: https://docs.lambda.ai/public-cloud/on-demand/ (fetched 2026-09-05)
Old URL https://lambdalabs.com/service/gpu-cloud returns HTTP 404.

Prices are per GPU per hour, but the instance is the billing unit: an 8x H100
instance costs 8 times its per-GPU rate, and you cannot rent a fraction of an
instance size that is not listed.

```
GPU                 VRAM     $/GPU-hr   Instance size   vCPU / RAM / SSD              Type
A6000               48 GB    1.09       1x              14 / 100 GiB / 512 GiB        on-demand
A6000               48 GB    1.09       2x              28 / 200 GiB / 1 TiB          on-demand
A6000               48 GB    1.09       4x              56 / 400 GiB / 1 TiB          on-demand
A10                 24 GB    1.29       1x              30 / 226 GiB / 1.3 TiB        on-demand
Quadro RTX 6000     24 GB    0.69       1x              14 / 46 GiB / 512 GiB         on-demand
A100 PCIe           40 GB    1.99       1x              30 / 225 GiB / 512 GiB        on-demand
A100 PCIe           40 GB    1.99       2x              60 / 450 GiB / 1 TiB          on-demand
A100 PCIe           40 GB    1.99       4x              120 / 900 GiB / 1 TiB         on-demand
A100 SXM            40 GB    1.99       1x              30 / 220 GiB / 512 GiB        on-demand
A100 SXM            40 GB    1.99       8x              124 / 1800 GiB / 5.8 TiB      on-demand
A100 SXM            80 GB    2.79       8x              240 / 1800 GiB / 19.5 TiB     on-demand
H100 PCIe           80 GB    3.29       1x              26 / 225 GiB / 1 TiB          on-demand
H100 SXM            80 GB    4.29       1x              26 / 225 GiB / 2.75 TiB       on-demand
H100 SXM            80 GB    4.19       2x              52 / 450 GiB / 5.5 TiB        on-demand
H100 SXM            80 GB    4.09       4x              104 / 900 GiB / 11 TiB        on-demand
H100 SXM            80 GB    3.99       8x              208 / 1800 GiB / 22 TiB       on-demand
GH200               96 GB    2.29       1x              64 / 432 GiB / 4 TiB          on-demand
B200 SXM6           180 GB   6.99       1x              26 / 360 GiB / 2.75 TiB       on-demand
B200 SXM6           180 GB   6.89       2x              52 / 720 GiB / 5.5 TiB        on-demand
B200 SXM6           180 GB   6.79       4x              104 / 1440 GiB / 11 TiB       on-demand
B200 SXM6           180 GB   6.69       8x              208 / 2900 GiB / 22 TiB       on-demand
Tesla V100          16 GB    0.79       8x              88 / 448 GiB / 5.8 TiB        on-demand
H200                --       NOT LISTED on the pricing page or the docs instance-type list
RTX 4090 / 5090, L40S, RTX 6000 Ada, RTX PRO 6000:   not offered
```

Lambda notes:

- Billing docs: on-demand instances are "billed in one-minute increments",
  from the moment the instance passes health checks until termination,
  "regardless if they're actively being used". No stop-and-keep state; a stopped
  instance is a terminated instance.
- No spot or preemptible tier exists.
- Storage: filesystems "billed per GiB used per month in one-hour increments";
  the docs example rate is "$0.20 per GiB per month", labeled as an illustration.
  Billing continues while the filesystem exists even if unmounted. The instance's
  local SSD in the table above is included in the hourly price.
- Egress: "Transparent pricing with no egress fees" on the page; docs say no
  charge for ingress or egress.
- H200: the page title advertises H200 but no H200 row exists in the table, and
  the docs instance-type list has no H200. A JarvisLabs blog (updated August 3,
  2026) mentions Lambda only for B200, not H200. No verifiable Lambda H200 rate.
- 1-Click Clusters (reserved, 2 weeks to 1 year, billed in weekly increments), as
  read by the fetch tool from lambda.ai/pricing:
  - B200: $9.86 / $9.36 / $8.87 per GPU-hr at 16 / 64 / 256+ GPUs
  - H100: $6.16 / $5.85 / $5.54 per GPU-hr at 16 / 64 / 256 GPUs
  These read HIGHER than on-demand, which is unusual; re-check the page before
  relying on them. They are irrelevant to a single-node drafter run anyway.

---

## 4. Together AI

URLs: https://www.together.ai/pricing and https://www.together.ai/gpu-clusters
(both fetched 2026-09-05). Page wording is "/hr per GPU".

```
GPU                 VRAM       On-demand $/GPU-hr   Reserved $/GPU-hr        Minimum scale
HGX H100            80 GB      3.99                 3.69 down to 3.19        8 to 256 GPUs
HGX H200            141 GB*    5.99                 4.99 down to 3.99        256 to 1,000 GPUs
HGX B200            180 GB     8.19                 7.99 down to 6.79        256 to 1,000+ GPUs
HGX B300            270 GB     contact              contact
GB200 NVL72         186 GB     contact              contact                  512 to 1,000+
GB300 NVL72         288 GB     contact              contact

Dedicated inference endpoint, H100:   3.99/hr   ("Promotion valid until 09/30/26")
Dedicated inference endpoint, B200:   8.99/hr
Dedicated inference endpoint, H200 / B300 / GB200 / GB300:   contact
```
*The gpu-clusters page states 140 GB for H200.

Together AI notes:

- Per GPU per hour, but the smallest cluster is 8 GPUs for H100 and 256 for
  H200 and B200. There is no single-GPU or 24 GB / 48 GB rental. Not a fit for
  this drafter run unless you want an 8x H100 node at 8 x $3.99 = $31.92/hr.
- Reserved tiers run 7 to 180+ days, paid upfront, up to 6 months. The reserved
  ranges above are the pricing page's "$3.69-$3.19 (H100), $4.99-$3.99 (H200),
  $7.99-$6.79 (B200) depending on commitment length."
- Storage: "Shared Filesystem $0.16 GiB/month".
- Egress: none listed. Spot: none. No consumer or workstation GPUs.

---

## 5. Quick view for the drafter run

Cheapest published on-demand price per VRAM class, single GPU, from the tables
above. Vast.ai values are today's verified-host minimum and are not a fixed rate.

```
Class    Card              RunPod Community   RunPod Secure   Vast min (on-demand)   Vast min (spot)   Lambda 1x   Together
24 GB    RTX 4090          0.34               0.74            0.223                  0.158             --          --
32 GB    RTX 5090          0.69               0.99            0.343                  not queried       --          --
48 GB    RTX A6000         0.33               0.53            0.402                  not queried       1.09        --
48 GB    RTX 6000 Ada      0.74               0.84            0.515                  not queried       --          --
48 GB    L40S              0.79               0.99            0.801                  not queried       --          --
80 GB    A100 PCIe         1.19               1.39            0.668 (40 GB card)     --                1.99 (40GB) --
80 GB    A100 SXM          1.39               1.59            0.564 (40 GB card)     0.401 (40 GB)     2.79 (8x)   --
80 GB    H100 PCIe         1.99               2.89            2.935                  not queried       3.29        --
80 GB    H100 SXM          2.69               3.29            2.406                  0.673             4.29        3.99 (8x min)
94 GB    H100 NVL          2.59               3.19            2.670                  not queried       --          --
96 GB    RTX PRO 6000      1.69               2.09            1.341                  not queried       --          --
141 GB   H200              3.59               4.59            3.977                  not queried       not listed  5.99 (256x min)
```

Spot summary:

- RunPod: spot exists per an August 25, 2026 blog post at roughly 50% off, but no
  spot rate is published on the pricing page or docs.
- Vast.ai: interruptible published live per offer; today's verified RTX 4090
  floor was $0.158 vs $0.223 on-demand, H100 SXM floor $0.673 vs $2.406.
- Lambda: no spot tier.
- Together AI: no spot tier.

Billing granularity summary:

- RunPod: per second, no egress, storage billed separately.
- Vast.ai: per second, storage and bandwidth billed separately at host-set rates.
- Lambda: per minute, no egress, local SSD included, filesystem extra.
- Together AI: per hour per GPU, cluster minimum 8 GPUs, storage extra.

Cross-check: GetDeploying's H200 page (as of September 5, 2026) shows RunPod H200
Community at $3.59, matching RunPod's page today, but shows Together H200 at
$2.99, which does not match Together's own page ($5.99). Trust the provider
pages over the aggregator.
