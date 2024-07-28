# Enhancing Point Annotations with Superpixel and Confidence Learning Guided for Improving Semi-Supervised OCT Fluid Segmentation

# Run SGPLG module to generate the pseudo-label

## data

链接：[https://pan.baidu.com/s/187KktO1_NMikH_sgU4LXEg](https://pan.baidu.com/s/187KktO1_NMikH_sgU4LXEg) 提取码：547m

```jsx
cd SGPLG/
python generate_label.py
```

# Training SGPA-Net

链接：[https://pan.baidu.com/s/1V0_qXcego09C8yvffe1F5Q](https://pan.baidu.com/s/1V0_qXcego09C8yvffe1F5Q)
提取码：ivht

```python
python SGPA.py
```

# Confident Learning generate refined-lables

```python
python middle.py
```

# Training SCLGPA-Net

```python
python SCLGPA.py
```

# Cition

```python
@article{weng2024enhancing,
  title={Enhancing point annotations with superpixel and confident learning guided for improving semi-supervised OCT fluid segmentation},
  author={Weng, Tengjin and Shen, Yang and Jin, Kai and Wang, Yaqi and Cheng, Zhiming and Li, Yunxiang and Zhang, Gewen and Wang, Shuai},
  journal={Biomedical Signal Processing and Control},
  volume={94},
  pages={106283},
  year={2024},
  publisher={Elsevier}
}
```