# coding:utf-8

from __future__ import unicode_literals

from keras.preprocessing.sequence import pad_sequences
# from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
from collections import Counter
from datetime import timedelta
from glob import glob
import numpy as np
import datetime
import codecs
import time
import os


def current_time():
    """
    获取当前时间：年月日时分秒
    :return:
    """
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def get_time(start_time):
    end_time = time.time()
    time_dif = start_time - end_time
    return timedelta(seconds=int(round(time_dif)))


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    X, Y = [u'\u4e00', u'\u9fa5']  # unicode 前面加u
    if uchar >= X and uchar <= Y:
        return True
    else:
        return False


def read_file(file_dirs, seq_length):
    """读取文件数据"""
    contents, labels = list(), list()
    for txt in file_dirs:
        try:
            with codecs.open(txt, 'r', encoding='utf-8') as fr:
                content = fr.read()
        except:
            print 'cannot open {} file, encoding error'.format(txt)
            continue
        new_content = list()
        for word in content:
            if is_chinese(word):
                new_content.append(word)
                if len(new_content) > seq_length - 1:
                    break
#        content = [word for word in content if is_chinese(word)][:seq_length]
        if len(new_content):
            contents.append(list(new_content))
            labels.append(txt.split('/')[-2])
    return contents, labels


def build_vocab(train_dirs, seq_length, vocab_dir='', vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dirs, seq_length)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    with codecs.open(vocab_dir, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(words) + '\n')
    # open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def read_category(categories):
    """读取分类目录，固定"""
    # categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def read_vocab(vocab_dir):
    """读取词汇表"""
    with codecs.open(vocab_dir, 'r', encoding='utf-8') as fr:
        words = fr.read()
    words = words.splitlines()
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_file(filename, seq_length, word_to_id, cat_to_id, max_length=400):
    """将文件转换为id表示"""
    contents, labels = read_file(filename, seq_length)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = pad_sequences(data_id, max_length)
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


if __name__ == '__main__':
    # types = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
    #          'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
    #          'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
    #          'T004019003', 'T004009001', 'T004009002', 'T004005001', 'T004005002', 'T004006001', 'T004006005',
    #          'OTHER']
    types = [
        '110100', '110200', '110300', '110400', '110500',
        '110600', '110700', '110800', '210100', '210200',
        '210300', '210400', '220100', '220200', '220300',
        '220400', '220500', '220600', '230100', '240200',
        '240300', '240400', '240500', '270100', '270200',
        '270300', '270400', '270500', '280100', '280200',
        '280300', '280400', '330100', '330200', '340300',
        '340400', '350100', '350200', '360100', '360200',
        '360300', '360400', '370100', '370200', '370300',
        '370400', '370500', '370600', '410100', '410200',
        '410300', '410400', '420100', '420200', '420300',
        '420400', '420500', '420600', '420700', '420800',
        '430100', '430200', '450200', '450300', '450400',
        '450500', '460100', '460200', '460300', '460400',
        '460500', '480100', '490100', '490200', '490300',
        '510100', '610100', '610200', '610300', '620100',
        '620200', '620300', '620400', '620500', '630100',
        '630200', '630300', '630400', '640100', '640200',
        '640300', '640400', '640500', '650100', '650200',
        '650300', '650400', '710100', '710200', '720100',
        '720200', '720300', '730100', '730200', '0'
    ]

    print "载入训练样本..."
    # data_dir = '/home/abc/ssd/pzw/nlp/data/0523/word_sep/'
    # data_dir = '/home/zhwpeng/abc/nlp/data/0324/word_sep/'
    # data_dir = '/home/zhwpeng/abc/text_classify/data/0412/raw/train_data/'
    data_dir = '/home/zhwpeng/abc/nlp/data/report_industry/'
    base_dir = 'vocab_1000/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    vocab_dir = os.path.join(base_dir, 'vocab.txt')

    txt_dirs = list()
    for fold in glob(data_dir + '*'):
        # txt_dirs = txt_dirs + glob(fold+'/*.txt')
        txt_dirs = txt_dirs + glob(fold + '/*.txt')[:4]  # 本地小批量数据
    print "训练样本总数是{}".format(len(txt_dirs))
    np.random.shuffle(txt_dirs)
    seq_length = 1000
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(txt_dirs, seq_length, vocab_dir=vocab_dir)
    words, word_to_id = read_vocab(vocab_dir=vocab_dir)
    # print words, len(words)
    print word_to_id, len(word_to_id)
    # for key in word_to_id.keys():
    #     print key
    # for i in range(10):
    #     print words[i], word_to_id[i]




