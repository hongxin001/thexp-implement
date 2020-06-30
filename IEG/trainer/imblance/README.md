## 4.1 MNIST data imbalance experiments


```bash
python run_imblance_base.py --device:cuda:0
python run_imblance_l2r.py --device:cuda:0
```

运行后可以看到结果差异十分明显。普通方法（base）在不平衡比例达到 0.995 的时候在测试集基本没有办法识别（保持在50%左右），而l2r可以达到93%-95%左右

