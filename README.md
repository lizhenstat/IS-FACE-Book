
## 📁 项目结构
```
IS-FACE-Book/
├── README.md                  # 项目简介
├── argsfile.py                # 超参数配置文件
├── all_loss.py                # 损失函数
├── backbones.py               # 构建用于特征提取的骨干网络结构
├── dataset.py                 # 构建训练数据加载
├── metrics.py                 # 性能评估指标（如准确率、TAR 等）
├── train_insightface_book.py  # 训练主函数（基于 InsightFace 改写）
├── utils_insightface.py       # 数据处理、模型保存等辅助工具函数
├── generate_meta/             # ⚙️ 生成 meta 数据（训练阶段所需的辅助信息）
│   ├── get_single_race_imgids.py     # 提取每个种族的训练样本标签与图像索引，生成四个种族对应的 pkl 文件
│   ├── RFW_get_imgid_label_dict.py   # 从 .rec 文件中提取图像 ID 与标签的映射关系，并保存为两个字典
│   └── RFW_get_label_race.py         # 构建 label → race 映射字典，用于种族划分
```


## 数据


## 命令行
`python train_insightface_book.py --RFW_race Caucasian --lr 0.1 --max_epoch 27  \
--clip_grad_norm --train_batch_size 128 --backbone r34 --metric arc_margin --arc_m 0.5`

## 📜 License

This project is licensed under the Apache License 2.0.

## 🔗 Attribution

This project partially refers to or modifies code from:

- The InsightFace project ([GitHub Link](https://github.com/deepinsight/insightface)) licensed under Apache 2.0.
- The RFW verification code `verification_RFW.py` provided by the authors of the RFW dataset ([Website](http://www.whdeng.cn/RFW/model.html)).

Please refer to their original papers for more details.
