import collections
import codecs
import os
import tensorflow as tf
import numpy as np

from bert import tokenization
from text_model import TextConfig



class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        构造bert模型样本的类
        Args:
          guid: 样本的编码，表示第几条数据，不是模型要输入的对应参数；
          text_a: 第一个序列文本，对应我们数据集要分类的文本；
          text_b: 第二个序列文本，是bert模型在sequence pair 任务要输入的文本，在我们这个场景不需要，设置为None;
          label: 文本标签
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class TextProcessor(object):
    """按照InputExample类形式载入对应的数据集"""

    """load train examples"""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train_3w.tsv")), "train")

    """load dev examples"""
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "dev_bert6.tsv")), "dev")

    """load test examples"""
    def get_test_examples(self, data_dir):
          return self._create_examples(
              self._read_file(os.path.join(data_dir, "test.tsv")), "test")

    """set labels"""
    def get_labels(self):
        return ['0', '1']

    """read file"""
    def _read_file(self, input_file):
        with codecs.open(input_file, "r",encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                try:
                    line=line.split('\t')
                    assert len(line)==3
                    lines.append(line)
                except:
                    pass
            np.random.shuffle(lines)
            return lines

    """create examples for the data set """
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
          guid = "%s-%s" % (set_type, i)
          text_a = tokenization.convert_to_unicode(line[0])
          text_b = tokenization.convert_to_unicode(line[1])
          label = tokenization.convert_to_unicode(line[2].strip())
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_examples_to_features(examples,label_list, max_seq_length,tokenizer):
    """
    将所有的InputExamples样本数据转化成模型要输入的token形式，最后输出bert模型需要的四个变量；
    input_ids：就是text_a(分类文本)在词库对应的token，按字符级；
    input_mask：bert模型mask训练的标记，都为1；
    segment_ids：句子标记，此场景只有text_a,都为0；
    label_ids：文本标签对应的token，不是one_hot的形式；
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_data=[]
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        features = collections.OrderedDict()
        features["input_ids"] = input_ids
        features["input_mask"] = input_mask
        features["segment_ids"] = segment_ids
        features["label_ids"] =label_id
        input_data.append(features)

    return input_data



def batch_iter(input_data,batch_size):
    """
    将样本的四个tokens形式的变量批量的输入给模型；
    """
    batch_ids,batch_mask,batch_segment,batch_label=[],[],[],[]
    for features in input_data:
        if len(batch_ids) == batch_size:
            yield batch_ids,batch_mask,batch_segment,batch_label
            batch_ids, batch_mask, batch_segment, batch_label = [], [], [], []

        batch_ids.append(features['input_ids'])
        batch_mask.append(features['input_mask'])
        batch_segment.append(features['segment_ids'])
        batch_label.append(features['label_ids'])

    if len(batch_ids) != 0:
        yield batch_ids, batch_mask, batch_segment, batch_label
