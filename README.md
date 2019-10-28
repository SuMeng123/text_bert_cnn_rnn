# TEXT_BERT_CNN_RNN
在 Google BERT Fine-tuning基础上，利用cnn或rnn进行中文文本的分类;<br>
<br>
本项目改编自[text_bert_cnn](https://github.com/cjymz886/text_bert_cnn)，原项目是一个cnn的10分类问题，我添加了Bi-Lstm可供大家选择，并且代码中已经改为2分类问题;<br>
<br>
没有使用tf.estimator API接口的方式实现，主要我不太熟悉，也不习惯这个API，还是按原先的[text_cnn](https://github.com/cjymz886/text-cnn)实现方式来的;<br>
<br>
训练结果：在验证集上准确率是96.4%左右，训练集是100%；，这个结果单独利用cnn也是可以达到的。这篇blog不是来显示效果如何，主要想展示下如何在bert的模型下Fine-tuning，觉得以后这种方式会成为主流。<br>

---
以下两个项目特别有参考价值，关于词向量嵌入、模型可视化、模型embedding、batch_iter等一系列流程，值得学习参考<br>
### [text_cnn项目](https://github.com/cjymz886/text-cnn)
嵌入词级别所做的CNN文本分类。本实验的主要目的是为了探究基于Word2vec训练的词向量嵌入CNN后，对模型的影响。
### [text-classification-cnn-rnn项目](https://github.com/gaussic/text-classification-cnn-rnn)
基于字符级别所做的CNN/RNN文本分类。<br>

1 环境
=
python3<br>
tensorflow 1.9.0以上

2 数据
=
还是以前的数据集，涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']；<br>
下载链接:[https://pan.baidu.com/s/11AuC5g47rnsancf6nfKdiQ] 密码:1vdg<br>

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip): Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

3 运行
=
python text_run.py train<br>
<br>
python text_run.py test<br>

4 结论
=
我个人感觉在bert基础上对text_cnn提升并不大，不过这个数据集我优化的最好结果在验证集上也只是97%左右，怀疑数据集中可能有些文本的类别不是特别明显，或是属于多个类别也是合理的<br>
<br>
bert在中文上目前只是支持字符级别的，而且文本长度最大为128(我怎么觉得是512？)，这个长度相对于单独卷积就处于劣势<br>
<br>
bert会导致运行效率降低很多，毕竟模型的参数量摆在那里，实际应用要一定的硬件支持<br>

5 参考
=

1. [google-research/bert](https://arxiv.org/abs/1408.5882)
2. [brightmart/bert_language_understanding](https://github.com/brightmart/bert_language_understanding)

6 在bert上层添加Bi-Lstm
=

当时自己在用原生bert，并没有想过在使用bert的预训练结果，在上层添加模型的方法，这是吕同学发给我的代码截图，我是第一次听说这种做法。发现这个代码很像自己的ner-slot_filling项目，可以看一下。

![吕给我的代码截图](./img/Image.png)

