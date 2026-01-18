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
## 预测模型构建

本项目采用三维卷积神经网络（3D CNN）对全脑 T1 MRI 进行端到端建模。  
模型采用多层编码结构，通过逐级下采样提取从局部解剖结构到全局空间模式的层级特征。

在网络设计上：
- 使用 3D 卷积以显式建模体素间空间关联
- 引入残差连接缓解深层网络训练不稳定问题
- 在编码器末端进行全局特征聚合，用于二分类判别

该设计能够在不依赖显式 ROI 标注的情况下，自动学习与帕金森病相关的结构性差异。

---
## 核心实现（Implementation）

### 数据预处理

- 脑区掩膜约束，去除非脑组织信号
- 基于样本自身的 Z-score 强度归一化
- 训练阶段引入三维空间与强度数据增强

```python
img = img * brain_mask
img = (img - img.mean()) / (img.std() + 1e-6)
