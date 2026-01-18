---
title: "基于三维 T1 磁共振影像的帕金森病自动诊断"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/parkinson-t1-cnn
date: 2026-01-18
excerpt: "基于常规 T1 MRI 与三维卷积神经网络，实现帕金森病的自动分类与模型性能评估。"
header:
  teaser: /images/portfolio/parkinson-t1-cnn/roc_curve.png
tags:
- 医学影像
- 深度学习
- 三维卷积神经网络
- 帕金森病
- 分类模型
tech_stack:
- name: Python
- name: PyTorch
- name: Scikit-learn
- name: NumPy
---

## 项目背景（Background）

帕金森病是一种常见的神经系统退行性疾病，具有起病隐匿、进展缓慢且不可逆的特点。随着病程进展，患者可出现运动功能障碍及多种非运动症状，严重影响生活质量。  
在临床实践中，帕金森病诊断主要依赖症状评估，缺乏客观、自动化的影像辅助诊断手段。

T1 加权磁共振成像作为最常规的脑部结构影像序列，具有获取方便、空间分辨率高、跨中心一致性较好的优势。本项目基于三维 T1 MRI，探索深度学习方法在帕金森病自动诊断中的应用可行性。

---

## 核心实现（Implementation）

### 数据预处理

- 脑区掩膜约束，去除非脑组织信号
- 基于样本自身的 Z-score 强度归一化
- 训练阶段引入三维空间与强度数据增强

```python
img = img * brain_mask
img = (img - img.mean()) / (img.std() + 1e-6)
