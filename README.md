# Reproduce AdderNet

Since in the [official repository](https://github.com/huawei-noah/AdderNet), the first author said they are [not releasing](https://github.com/huawei-noah/AdderNet/issues/2) this training code while a few people are looking for it. I'm releasing this reproduction of AdderNet results. (It's been a month tho)

The original paper is available on [arXiv](https://arxiv.org/pdf/1912.13200.pdf).

## Known issues

The researchers introduced an adaptive learning rate scaling coefficient (which can be found in section 3.3) to amplify gradients, and one of the components in the formula is a scaler denoted by ita. Along with the experiment results the ita values (adder layer coefficient) were not given. Because I have limited computation resources I am not attempting to look for its best setting. And certainly I wasn't able to reproduce the result. If anyone managed to find the optima please feel free to open a pull request. >.<"

# How to Use

```
python ./run.py
```

# License

This code repo is under [MIT License](LICENSE).
