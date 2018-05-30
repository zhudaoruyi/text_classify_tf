# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100  # 词向量维度
    seq_length = 1000  # 序列长度
    num_classes = 105  # 类别数
    # num_filters = 256  # 卷积核数目
    num_filters = [256, 128, 64]  # 卷积核数目
    # kernel_size = 5  # 卷积核尺寸
    filter_sizes = [3, 5, 7, 9, 11]
    vocab_size = 5000  # 词汇表达
    # vocab_size = 97639

    hidden_dim = 512  # 全连接层神经元

    dropout_keep_prob = 1.  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 30  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 100  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        pooling_merged = list()
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope('conv_maxpool_%s' % i):
                conv_1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters[0], filter_size,
                                           name='conv_1%s' % i)
                h_1 = tf.nn.relu(conv_1, name='relu_1%s' % i)
                pooling_1 = tf.nn.pool(h_1, window_shape=[3], pooling_type="MAX", strides=[1], padding="SAME", 
                                       name='pool_1%s' % i)
                
                conv_2 = tf.layers.conv1d(pooling_1, self.config.num_filters[1], 4, name='conv_2%s' % i)
                h_2 = tf.nn.relu(conv_2, name='relu_2%s' % i)
                pooling_2 = tf.nn.pool(h_2, window_shape=[3], pooling_type="MAX", strides=[1], padding="SAME", 
                                       name='pool_2%s' % i)

                conv_3 = tf.layers.conv1d(pooling_2, self.config.num_filters[2], 6, name='conv_3%s' % i)
                h_3 = tf.nn.relu(conv_3, name='relu_3%s' % i)
                pooling_3 = tf.reduce_max(h_3, reduction_indices=[1], name='global_max_pooling_%s' % i)
                pooling_merged.append(pooling_3)

        # with tf.name_scope("conv_maxpool_0"):
        #     conv_31 = tf.layers.conv1d(embedding_inputs, self.config.num_filters[0], self.config.filter_sizes[0], name='conv_31')
        #     h_31 = tf.nn.relu(conv_31, name='relu')
        #     pooling_31 = tf.nn.pool(h_31,window_shape=[3], pooling_type="MAX", strides=[1], name="pool31", padding="SAME")
        #     conv_32 = tf.layers.conv1d(pooling_31, self.config.num_filters[1], 4, name='conv_32')
        #     h_32 = tf.nn.relu(conv_32, name='relu')
        #     pooling_32 = tf.nn.pool(h_32, window_shape=[3], pooling_type="MAX",strides=[1], name="pool32", padding="SAME")
        # 
        #     conv_33 = tf.layers.conv1d(pooling_32, self.config.num_filters[2], 6, name='conv_33')
        #     h_33 = tf.nn.relu(conv_33, name='relu')
        #     pooling_33 = tf.reduce_max(h_33, reduction_indices=[1], name='global_max_pooling')
        # 
        # with tf.name_scope("conv_maxpool_1"):
        #     conv_51 = tf.layers.conv1d(embedding_inputs, self.config.num_filters[0], self.config.filter_sizes[0],
        #                                name='conv_51')
        #     h_51 = tf.nn.relu(conv_51, name='relu')
        #     pooling_51 = tf.nn.pool(h_51, window_shape=[3], pooling_type="MAX", strides=[1], name="pool51", padding="SAME")
        #     conv_52 = tf.layers.conv1d(pooling_51, self.config.num_filters[1], 4, name='conv_52')
        #     h_52 = tf.nn.relu(conv_52, name='relu')
        #     pooling_52 = tf.nn.pool(h_52, window_shape=[3],pooling_type="MAX",strides=[1], name="pool52",padding="SAME")
        # 
        #     conv_53 = tf.layers.conv1d(pooling_52, self.config.num_filters[2], 6, name='conv_53')
        #     h_53 = tf.nn.relu(conv_53, name='relu')
        #     pooling_53 = tf.reduce_max(h_53, reduction_indices=[1], name='global_max_pooling')
        # 
        # with tf.name_scope("conv_maxpool_2"):
        #     conv_71 = tf.layers.conv1d(embedding_inputs, self.config.num_filters[0], self.config.filter_sizes[0],
        #                                name='conv_71')
        #     h_71 = tf.nn.relu(conv_71, name='relu')
        #     pooling_71 = tf.nn.pool(h_71, window_shape=[3],pooling_type="MAX",strides=[1], name="pool71",padding="SAME")
        #     conv_72 = tf.layers.conv1d(pooling_71, self.config.num_filters[1], 4, name='conv_72')
        #     h_72 = tf.nn.relu(conv_72, name='relu')
        #     pooling_72 = tf.nn.pool(h_72, window_shape=[3],pooling_type="MAX",strides=[1], name="pool72",padding="SAME")
        #     conv_73 = tf.layers.conv1d(pooling_72, self.config.num_filters[2], 6, name='conv_73')
        #     h_73 = tf.nn.relu(conv_73, name='relu')
        #     pooling_73 = tf.reduce_max(h_73, reduction_indices=[1], name='global_max_pooling')
        # 
        # num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        # pooled_concat = tf.concat([pooling_33, pooling_53, pooling_73], axis=1, name='pooled_concat')
        pooled_concat = tf.concat(pooling_merged, axis=1, name='pooled_concat')
        # pooled_concat_flat = tf.reshape(pooled_concat, shape=[-1, num_filters_total], name='pooled_concat_flat')

        with tf.name_scope("dense"):
            # 全连接层，后面接dropout以及relu激活
            # fc = tf.layers.dense(self.pooled_concat, self.config.hidden_dim, name='fc1')
            fc = tf.layers.dense(pooled_concat, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.score = tf.nn.softmax(self.logits, name='score')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
