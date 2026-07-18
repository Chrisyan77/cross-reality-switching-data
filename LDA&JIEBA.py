# -*- coding: utf-8 -*-
import os
import time
import re
import numpy as np
import jieba
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text

# 设定中文字体，避免中文Glyph缺失
matplotlib.rc('font', family='Microsoft YaHei')
matplotlib.rc('axes', unicode_minus=False)


###############################################################################
# A. 停用词 / 分词
###############################################################################
def load_stopwords(filepath):
    """从本地文件加载停用词，返回一个 set。"""
    stopwords_set = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords_set.add(w)
    return stopwords_set


def preprocess_texts(docs, stopwords_set):
    """
    正则去除英数字 -> jieba分词 -> 去停用词
    返回: [ [词1, 词2, ...], [词1, 词2, ...], ... ]
    """
    processed = []
    for doc in docs:
        # 1) 去英数字
        seg_lists = re.sub(r"[A-Za-z0-9]", "", doc)
        # 2) 分词
        seg_list = jieba.cut(seg_lists)
        # 3) 过滤停用词 & 空白
        filtered = [w.strip() for w in seg_list if w.strip() and w not in stopwords_set]
        processed.append(filtered)
    return processed


###############################################################################
# B. LDA主题 / 词向量 / 相似词
###############################################################################
def estimate_topic_prior(lda_model, corpus):
    """
    估计 p(topic=k)：主题在语料中的先验概率
    通过遍历所有文档的主题分布(doc2bow)，最后取平均。
    """
    topic_counts = np.zeros(lda_model.num_topics, dtype=float)
    total_docs = 0
    for doc_bow in corpus:
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        for t_id, prob in doc_topics:
            topic_counts[t_id] += prob
        total_docs += 1
    p_topic = topic_counts / max(total_docs, 1)
    return p_topic


def word_topic_vector(lda_model, word_id, p_topic, topics_matrix):
    """
    对单词 word_id 计算在K个主题上的分布: v_w[k] = p(topic=k|word)
    p(k|w) ∝ p(w|k)*p(k)，再归一化。
    """
    p_w_given_t = topics_matrix[:, word_id]  # shape=(num_topics,)
    numerator = p_w_given_t * p_topic
    denom = numerator.sum()
    if denom < 1e-15:
        return np.zeros_like(p_topic)
    return numerator / denom


def build_word_vectors(lda_model, dictionary, corpus):
    """
    构造 {词: LDA主题向量} 用于后续相似词检索。
    v_w = p(topic|word)
    """
    p_topic = estimate_topic_prior(lda_model, corpus)
    topics_matrix = lda_model.get_topics()  # (num_topics, vocab_size)
    word_vectors = {}
    for w, w_id in dictionary.token2id.items():
        vec = word_topic_vector(lda_model, w_id, p_topic, topics_matrix)
        word_vectors[w] = vec
    return word_vectors


def find_similar_words_with_freq(target_word, word_vectors, dictionary,
                                 topn=10, alpha=0.8, freq_threshold=0):
    """
    同时考虑:
      (1) LDA主题相似度: sim
      (2) 词文档频次: freq_norm = doc_freq / max_doc_freq
    综合得分 = alpha*sim + (1-alpha)*freq_norm

    freq_threshold>0时可过滤文档出现数<freq_threshold的词。
    """
    if target_word not in word_vectors:
        return []

    target_vec = word_vectors[target_word]
    target_norm = np.linalg.norm(target_vec)
    if target_norm < 1e-15:
        return []

    freq_map = dictionary.dfs  # {token_id: doc_freq}
    max_freq = max(freq_map.values()) if freq_map else 1

    results = []
    target_id = dictionary.token2id.get(target_word, None)

    for w, vec in word_vectors.items():
        if w == target_word:
            continue

        w_id = dictionary.token2id[w]
        w_docfreq = freq_map.get(w_id, 0)
        if freq_threshold > 0 and w_docfreq < freq_threshold:
            continue

        # (A) 主题相似度
        denom = target_norm * np.linalg.norm(vec)
        sim = 0.0 if denom < 1e-15 else np.dot(target_vec, vec) / denom

        # (B) freq_norm
        freq_norm = w_docfreq / max_freq

        # (C) 综合得分
        score = alpha*sim + (1-alpha)*freq_norm
        results.append((w, sim, w_docfreq, score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:topn]


###############################################################################
# C. 2D可视化 (PCA/TSNE + matplotlib + adjustText)
###############################################################################
def visualize_words_2d(subset_word_vectors, highlight_words=None, title="LDA Word Vectors"):
    """
    将 subset_word_vectors 的主题向量 (K维) 降维到2D，并使用 adjustText
    自动排版以避免文字重叠。highlight_words可高亮显示关键词。
    """
    if not subset_word_vectors:
        print("没有可视化的词汇。")
        return

    words = list(subset_word_vectors.keys())
    vectors = np.array([subset_word_vectors[w] for w in words])

    # 1) PCA降维
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(vectors)

    # 2) 创建画布
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    ax.set_title(title, fontsize=16)

    # 3) 绘制散点
    ax.scatter(coords_2d[:,0], coords_2d[:,1], c="grey", alpha=1, edgecolors="white")

    # 4) 创建文本对象
    text_objs = []
    for i, w in enumerate(words):
        x, y = coords_2d[i, 0], coords_2d[i, 1]
        txt = ax.text(x, y, w, fontsize=12, color='grey')
        text_objs.append(txt)

    # 如果需要高亮某些词
    if highlight_words:
        for hw in highlight_words:
            if hw in words:
                idx = words.index(hw)
                x, y = coords_2d[idx,0], coords_2d[idx,1]
                # 高亮散点
                ax.scatter([x],[y], s=120, c="black", edgecolors="White")
                # 调整文本对象样式
                text_objs[idx].set_fontsize(14)
                text_objs[idx].set_fontweight("bold")
                text_objs[idx].set_color("black")
 
    # 5) 使用 adjustText 自动调整，避免文字重叠
    adjust_text(
        text_objs,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color='gray', alpha=0.1),
        expand_text=(2, 2),
        expand_points=(2, 2),
    )

    plt.show()


###############################################################################
# D. 主流程
###############################################################################
if __name__ == "__main__":
    #-------------------------
    # 1) 停用词
    #-------------------------
    stopwords_path = "D:/DeskTop/毕业论文/知网停用词.txt"
    if os.path.exists(stopwords_path):
        stopwords = load_stopwords(stopwords_path)
        print(f"已加载停用词表：{stopwords_path}")
    else:
        stopwords = {"的", "了", "和", "是", "也", "我", "在", "对", "中", "如果", "没有", "visionOS"}
        print("提示：未找到停用词文件，使用了少量内置停用词。")

    #-------------------------
    # 2) 读取文本
    #-------------------------
    input_file = "D:/DeskTop/only中文摘要.txt"
    if not os.path.exists(input_file):
        print(f"错误：找不到 {input_file} 文件！请确认路径。")
        exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        raw_docs = [line.strip() for line in f if line.strip()]

    if not raw_docs:
        print("提示：文本为空，结束。")
        exit(0)
    print(f"共读取到 {len(raw_docs)} 行文档。")

    #-------------------------
    # 3) 分词 + 去停用词
    #-------------------------
    start_time = time.time()
    processed_docs = preprocess_texts(raw_docs, stopwords)
    print(f"分词完成，耗时 {time.time() - start_time:.2f} 秒。")

    # (可选) 写入文件
    output_file = "D:/DeskTop/cih.txt"
    with open(output_file, "w", encoding="utf-8") as out_f:
        for tokens in processed_docs:
            out_f.write(" ".join(tokens) + "\n")
    print(f"已将分词结果写入 {output_file}")

    #-------------------------
    # 4) LDA 训练
    #-------------------------
    dictionary = corpora.Dictionary(processed_docs)
    corpus_bow = [dictionary.doc2bow(doc) for doc in processed_docs]

    num_topics = 2
    lda_model = models.ldamodel.LdaModel(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=5,
        iterations=100
    )
    print("\n=============================")
    print("  LDA 主题分析：")
    print("=============================")
    for topic_id in range(num_topics):
        top_terms = lda_model.print_topic(topic_id, topn=10)
        print(f"主题 {topic_id} : {top_terms}")

    # 示例：查看第一篇文档的主题分布
    if processed_docs:
        doc_index = 0
        doc_topics = lda_model.get_document_topics(corpus_bow[doc_index])
        print(f"\n示例：第1篇文档的主题分布: {doc_topics}")

    #-------------------------
    # 5) 构建词向量 + 计算相似
    #-------------------------
    word_vecs = build_word_vectors(lda_model, dictionary, corpus_bow)

    # 以 "用户","体验" 为测试关键词
    target_words = ["用户", "感知","体验"]
    for kw in target_words:
        if kw not in word_vecs:
            print(f"【{kw}】不在词典中，跳过。")
            continue

        # 查找相似词 (含词频 + 主题相似)
        results = find_similar_words_with_freq(
            target_word=kw,
            word_vectors=word_vecs,
            dictionary=dictionary,
            topn=20,
            alpha=0.95,
            freq_threshold=1  # 过滤出现文档数<2的词
        )
        print(f"\n与『{kw}』最相关的Top20 (主题+词频):")
        for w, simv, freqv, scorev in results:
            print(f"{w}\t相似度={simv:.3f}\t出现文档数={freqv}\t综合得分={scorev:.3f}")

    # (可选) 2D可视化
    # 收集关键词 + 相似词
    words_to_show = set()
    for kw in target_words:
        if kw in word_vecs:
            words_to_show.add(kw)
            sub = find_similar_words_with_freq(kw, word_vecs, dictionary, topn=20, alpha=0.80, freq_threshold=1)
            for w, simv, freqv, scv in sub:
                words_to_show.add(w)

    subset_vecs = {w: word_vecs[w] for w in words_to_show if w in word_vecs}
    if len(subset_vecs) > 2:
        visualize_words_2d(subset_vecs, highlight_words=target_words, title="知网摘要中用户感知相关LDA主题向量可视化图表")
    else:
        print("2D可视化词汇太少，跳过。")

    #-------------------------
    # 6) pyLDAvis
    #-------------------------
    print("\n开始 pyLDAvis 可视化...")
    vis_data = gensimvis.prepare(lda_model, corpus_bow, dictionary)
    vis_file = "D:/DeskTop/lda.html"
    pyLDAvis.save_html(vis_data, vis_file)
    print(f"已保存到 {vis_file}，用浏览器打开可查看。")