# 这是一个学习项目
- 将ray的04_finetuning_llms_with_deepspeed 重新改造会原生accelerate + deepspeed 使用 llama 3.2 1B模型，在本地8G显卡+offload cpu上完成训练
- 参考的另一个项目为https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
- 其次为https://huggingface.co/learn/nlp-course

# 一些学习心得
- 04_finetuning_llms_with_deepspeed从accelerate examples改造来的时候，在数据集处理方面有改动，特别是最终的datasets需要包含labels列
- 在小容量显卡运行时，需要及时开启 accelerate + deepspeed zero3 offload cpu能力
