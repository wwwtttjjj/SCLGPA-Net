# Enhancing Point Annotations with Superpixel and Confidence Learning Guided for Improving Semi-Supervised OCT Fluid Segmentation

# Run SGPLG module to generate the pseudo-label

## data

链接：[https://pan.baidu.com/s/187KktO1_NMikH_sgU4LXEg](https://pan.baidu.com/s/187KktO1_NMikH_sgU4LXEg) 提取码：547m

```jsx
cd SGPLG/
python generate_label.py
```

# Training SGPA-Net(需要先把对应的数据下载放入主目录)

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
@misc{weng2023enhancing,
      title={Enhancing Point Annotations with Superpixel and Confidence Learning Guided for Improving Semi-Supervised OCT Fluid Segmentation}, 
      author={Tengjin Weng and Yang Shen and Kai Jin and Zhiming Cheng and Yunxiang Li and Gewen Zhang and Shuai Wang and Yaqi Wang},
      year={2023},
      eprint={2306.02582},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```