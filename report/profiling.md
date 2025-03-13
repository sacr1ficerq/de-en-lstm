## Оптимизация teacher forcing

| Name                                                                        | Self CPU % | Self CPU  | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| --------------------------------------------------------------------------- | ---------- | --------- | ----------- | --------- | ------------ | --------- | ----------- | ---------- | ------------- | ---------- |
| `aten::copy_`                                                             | 28.86%     | 5.878s    | 28.86%      | 5.878s    | 133.093us    | 10.713s   | 46.42%      | 10.713s    | 242.559us     | 44166      |
| `aten::to`                                                                | 0.11%      | 21.962ms  | 27.01%      | 5.502s    | 603.553us    | 19.197ms  | 0.08%       | 6.505s     | 713.577us     | 9116       |
| `aten::_to_copy`                                                          | 0.23%      | 46.282ms  | 26.91%      | 5.480s    | 2.149ms      | 11.916ms  | 0.05%       | 6.486s     | 2.543ms       | 2550       |
| `autograd::engine::evaluate_function: struct torch::autograd::CopySlices` | 0.19%      | 39.422ms  | 2.11%       | 429.041ms | 191.195us    | 7.573ms   | 0.03%       | 3.998s     | 1.782ms       | 2244       |
| `struct torch::autograd::CopySlices`                                      | 0.49%      | 100.022ms | 1.91%       | 389.619ms | 173.627us    | 31.028ms  | 0.13%       | 3.991s     | 1.778ms       | 2244       |
| `aten::linear`                                                            | 0.35%      | 71.396ms  | 1.58%       | 321.578ms | 116.768us    | 21.630ms  | 0.09%       | 2.392s     | 868.514us     | 2754       |
| `aten::addmm`                                                             | 0.62%      | 125.736ms | 0.67%       | 137.476ms | 51.839us     | 2.319s    | 10.05%      | 2.323s     | 876.102us     | 2652       |
| `autograd::engine::evaluate_function: AddmmBackward0`                     | 0.53%      | 108.503ms | 2.53%       | 515.485ms | 194.376us    | 22.751ms  | 0.10%       | 1.872s     | 706.038us     | 2652       |
| `AddmmBackward0`                                                          | 0.49%      | 99.115ms  | 1.61%       | 327.481ms | 123.485us    | 30.174ms  | 0.13%       | 1.684s     | 634.888us     | 2652       |
| `autograd::engine::evaluate_function: CudnnRnnBackward`                   | 1.06%      | 215.889ms | 16.16%      | 3.292s    | 1.403ms      | 38.598ms  | 0.17%       | 1.665s     | 709.835us     | 2346       |

---

Для сравнения таблица для обычного форварда:

| Name                                                              | Self CPU % | Self CPU  | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| ----------------------------------------------------------------- | ---------- | --------- | ----------- | --------- | ------------ | --------- | ----------- | ---------- | ------------- | ---------- |
| `autograd::engine::evaluate_function: AddmmBackward0`           | 0.33%      | 16.051ms  | 1.90%       | 93.427ms  | 183.190us    | 3.200ms   | 0.06%       | 938.012ms  | 1.839ms       | 510        |
| `AddmmBackward0`                                                | 0.41%      | 19.957ms  | 1.36%       | 66.563ms  | 130.516us    | 5.264ms   | 0.10%       | 820.590ms  | 1.609ms       | 510        |
| `aten::mm`                                                      | 0.63%      | 30.851ms  | 0.63%       | 30.851ms  | 23.267us     | 818.848ms | 15.57%      | 818.848ms  | 617.532us     | 1326       |
| `aten::linear`                                                  | 0.24%      | 11.580ms  | 1.50%       | 73.622ms  | 120.297us    | 3.487ms   | 0.07%       | 727.171ms  | 1.188ms       | 612        |
| `enumerate(DataLoader)#_SingleProcessDataLoaderIter._next_data` | 1.04%      | 51.007ms  | 29.59%      | 1.453s    | 14.244ms     | 594.000us | 0.01%       | 712.701ms  | 6.987ms       | 102        |
| `aten::pad_sequence`                                            | 10.67%     | 524.009ms | 28.46%      | 1.398s    | 6.851ms      | 145.432ms | 2.76%       | 712.075ms  | 3.491ms       | 204        |
| `aten::addmm`                                                   | 0.47%      | 23.056ms  | 0.50%       | 24.696ms  | 48.423us     | 704.618ms | 13.40%      | 705.381ms  | 1.383ms       | 510        |
| `autograd::engine::evaluate_function: CudnnRnnBackward`         | 0.15%      | 7.512ms   | 7.63%       | 374.743ms | 1.837ms      | 648.000us | 0.01%       | 570.837ms  | 2.798ms       | 204        |
| `CudnnRnnBackward0`                                             | 0.10%      | 5.078ms   | 7.48%       | 367.231ms | 1.800ms      | 637.000us | 0.01%       | 570.189ms  | 2.795ms       | 204        |
| `aten::_cudnn_rnn_backward`                                     | 5.86%      | 287.620ms | 7.38%       | 362.154ms | 1.775ms      | 508.806ms | 9.67%       | 569.552ms  | 2.792ms       | 204        |
