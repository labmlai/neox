# Half-Precision

```json
{
  "results": {
    "anli_r1": {
      "acc": 0.336,
      "acc_stderr": 0.014944140233795025
    },
    "anli_r2": {
      "acc": 0.335,
      "acc_stderr": 0.014933117490932573
    },
    "anli_r3": {
      "acc": 0.3525,
      "acc_stderr": 0.01379716491891836
    },
    "hellaswag": {
      "acc": 0.5352519418442542,
      "acc_stderr": 0.004977364364795599,
      "acc_norm": 0.7144991037641903,
      "acc_norm_stderr": 0.004507296196227817
    },
    "lambada": {
      "ppl": 3.6375113870173443,
      "ppl_stderr": 0.07468955218711418,
      "acc": 0.7203570735493887,
      "acc_stderr": 0.006252992453443255
    },
    "piqa": {
      "acc": 0.7758433079434167,
      "acc_stderr": 0.009729897956410041,
      "acc_norm": 0.7845484221980413,
      "acc_norm_stderr": 0.00959246311565811
    },
    "winogrande": {
      "acc": 0.6582478295185478,
      "acc_stderr": 0.013330103018622863
    },
    "wsc": {
      "acc": 0.5,
      "acc_stderr": 0.04926646390821466
    },
    "mathqa": {
      "acc": 0.2733668341708543,
      "acc_stderr": 0.008158890612550701,
      "acc_norm": 0.2720268006700168,
      "acc_norm_stderr": 0.008146370028043097
    }
  },
  "versions": {
    "anli_r1": 0,
    "anli_r2": 0,
    "anli_r3": 0,
    "hellaswag": 0,
    "lambada": 0,
    "piqa": 0,
    "winogrande": 0,
    "wsc": 0,
    "mathqa": 0
  },
  "config": {
    "name": "half_precision"
  }
}
```

| Task       | Metric          | NeoX Impl (2 GPU) | zphang (1 GPU)   | This repo (1 GPU) |
|------------|-----------------|-------------------|------------------|-------------------|
| anli_r1    | acc             | 0.3270            | 0.3300           | 0.3360            |
|            | acc_stderr      | 0.0148            | 0.0149           | 0.0149            | 
| anli_r2    | acc             | 0.3410            | 0.3420           | 0.3350            |
|            | acc_stderr      | 0.0150            | 0.0150           | 0.0149            |
| anli_r3    | acc             | 0.3567            | 0.3617           | 0.3525            |
|            | acc_stderr      | 0.0138            | 0.0139           | 0.0149            |
| hellaswag  | acc             | 0.5351            | 0.5335           | 0.5353            |
|            | acc_stderr      | 0.0050            | 0.0050           | 0.0050            |
|            | acc_norm        | 0.7140            | 0.7126           | 0.7145            |
|            | acc_norm_stderr | 0.0045            | 0.0045           | 0.0045            |
| lambada    | acc             | 0.7211            | 0.7223           | 0.7204            |
|            | acc_stderr      | 0.0062            | 0.0062           | 0.0063            |
|            | ppl             | 3.6760            | 3.6559           | 3.6375            |
|            | ppl_stderr      | 0.0760            | 0.0757           | 0.0747            |
| piqa       | acc             | 0.7748            | 0.7758           | 0.7758            |
|            | acc_stderr      | 0.0097            | 0.0097           | 0.0097            |
|            | acc_norm        | 0.7786            | 0.7856           | 0.7845            |
|            | acc_norm_stderr | 0.0097            | 0.0096           | 0.0096            |
| winogrande | acc             | 0.6598            | 0.6598           | 0.6582            |
|            | acc_stderr      | 0.0133            | 0.0133           | 0.0133            |
| wsc        | acc             | 0.5096            | 0.4808           | 0.5000            |
|            | acc_stderr      | 0.0493            | 0.0492           | 0.0493            |
