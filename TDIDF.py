# -*- coding: utf-8 -*-
import os
import re
import time
import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Gensim & PyLDAvis
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

###################################################
# 1) 读摘要 + NLTK分词 + 停用词 + TF-IDF过滤
###################################################
def load_stopwords(filepath):
    """
    从本地文件加载停用词 (每行一个)
    """
    stops = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                stops.add(w)
    return stops

def simple_tokenize(text, stops):
    """
    英文文本最简单分词 + 去除停用词
    - 全部转小写
    - 筛选纯字母
    - 剔除停用词
    """
    # 小写
    txt_lower = text.lower()
    # 用正则提取英文单词
    tokens = re.findall(r'[a-z]+', txt_lower)
    # 去掉停用词
    filtered = [w for w in tokens if w not in stops]
    return filtered

def tfidf_filter(processed_docs, min_tfidf=0.05):
    """
    先将每篇文档(分词后)拼成字符串, 用 TfidfVectorizer 计算
    对所有词取“平均TF-IDF”，保留 >= min_tfidf 的词
    """
    docs_str = [" ".join(doc) for doc in processed_docs]

    vectorizer = TfidfVectorizer(min_df=2, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(docs_str)
    feature_names = vectorizer.get_feature_names_out()

    mean_scores = tfidf_matrix.mean(axis=0).A1
    selected_words = {feature_names[i] for i in range(len(feature_names)) if mean_scores[i]>= min_tfidf}

    filtered = []
    for doc in processed_docs:
        new_doc = [w for w in doc if w in selected_words]
        filtered.append(new_doc)

    print(f"TF-IDF过滤后, 词典剩余 {len(selected_words)} 个 (阈值={min_tfidf}).")
    return filtered

###################################################
# 2) LDA & 相似词
###################################################
def estimate_topic_prior(lda_model, corpus):
    """
    估计 p(topic=k) => 主题在语料中的先验分布
    """
    topic_counts = np.zeros(lda_model.num_topics, dtype=float)
    total_docs = 0
    for doc_bow in corpus:
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        for t_id, prob in doc_topics:
            topic_counts[t_id] += prob
        total_docs += 1
    return topic_counts / max(total_docs, 1)

def word_topic_vector(lda_model, word_id, p_topic, topics_matrix):
    """
    v_w[k] = p(topic=k|word) ~ p(w|topic=k)*p(topic=k)
    """
    p_w_given_t = topics_matrix[:, word_id]
    numerator = p_w_given_t * p_topic
    denom = numerator.sum()
    if denom<1e-15:
        return np.zeros_like(p_topic)
    return numerator/denom

def build_word_vectors(lda_model, dictionary, corpus):
    """
    为每个词(在dictionary中)构建主题向量 => {词: 向量}
    """
    p_topic = estimate_topic_prior(lda_model, corpus)
    topics_matrix = lda_model.get_topics()  # shape=(num_topics,vocab_size)
    word_vecs = {}
    for w, w_id in dictionary.token2id.items():
        vec = word_topic_vector(lda_model, w_id, p_topic, topics_matrix)
        word_vecs[w] = vec
    return word_vecs

def find_similar_words_with_freq(target_word, word_vectors, dictionary, topn=10, alpha=0.8, freq_threshold=2):
    """
    同时考虑LDA相似度 + 词文档频次 => alpha*sim + (1-alpha)*(freq_norm)
    """
    if target_word not in word_vectors:
        return []

    target_vec = word_vectors[target_word]
    target_norm = np.linalg.norm(target_vec)
    if target_norm<1e-15:
        return []

    freq_map = dictionary.dfs  # {token_id: doc_freq}
    max_freq = max(freq_map.values()) if freq_map else 1

    results = []
    for w, vec in word_vectors.items():
        if w == target_word:
            continue
        w_id = dictionary.token2id[w]
        w_freq = freq_map.get(w_id,0)
        if w_freq < freq_threshold:
            continue
        denom = target_norm * np.linalg.norm(vec)
        sim = 0.0 if denom<1e-15 else np.dot(target_vec, vec)/denom

        freq_norm = w_freq/max_freq
        score = alpha*sim + (1-alpha)*freq_norm
        results.append((w, sim, w_freq, score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:topn]

###################################################
# main
###################################################
def main():
    # ============ 1) 读取only摘要&分词&去停用词 =============
    input_file = "d:/DeskTop/only摘要.txt"
    stop_file  = "d:/DeskTop/stop.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} 不存在!")
        return
    if not os.path.exists(stop_file):
        print(f"Error: {stop_file} 不存在!")
        return

    with open(input_file, "r", encoding="utf-8") as fin:
        raw_abstracts = [line.strip() for line in fin if line.strip()]

    # nltk stop
    nltk.download('punkt', quiet=True)
    # 也可下载 nltk.corpus.stopwords english
    local_stops = load_stopwords(stop_file)

    # 分词 + 去停用词
    processed_docs = []

    for abs_text in raw_abstracts:
        tokens = simple_tokenize(abs_text, local_stops)
        if tokens:
            processed_docs.append(tokens)

    print(f"共有 {len(processed_docs)} 篇摘要, 分词去停用词后开始 TF-IDF过滤...")
    # ============ 2) TF-IDF过滤 =============
    filtered_docs = tfidf_filter(processed_docs, min_tfidf=0.0015)

    # ============ 3) LDA训练 =============
    dictionary = corpora.Dictionary(filtered_docs)
    corpus_bow = [dictionary.doc2bow(doc) for doc in filtered_docs]

    # 你可根据需要调整 num_topics、passes等
    num_topics = 3
    lda_model = models.LdaModel(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=5,
        iterations=100
    )
    print("\n=== LDA topics ===")
    for k in range(num_topics):
        terms = lda_model.print_topic(k, topn=10)
        print(f"Topic {k}: {terms}")

    # ============ 4) 构建向量 + 相似查找 =============
    word_vecs = build_word_vectors(lda_model, dictionary, corpus_bow)

    # 需要分析与 "user","experience","perception" 三词相似的 top10
    targets = ["user","experience","perception"]
    for kw in targets:
        if kw not in word_vecs:
            print(f"\n词 '{kw}' 不在字典中, 无法找相似.")
            continue
        res = find_similar_words_with_freq(
            target_word=kw,
            word_vectors=word_vecs,
            dictionary=dictionary,
            topn=10,
            alpha=0.95,
            freq_threshold=2
        )
        print(f"\n与『{kw}』最相似的词 Top10:")
        for w, simv, freqv, scorev in res:
            print(f"{w}\tSim={simv:.3f}\tfreq={freqv}\tscore={scorev:.3f}")

    # ============ 5) 交互式可视化(可选) =============
    # 仅示例: pyLDAvis
    print("\n开始 pyLDAvis 可视化 (可选)...")
    try:
        vis_data = gensimvis.prepare(lda_model, corpus_bow, dictionary)
        out_html = "d:/DeskTop/lda_vis.html"
        pyLDAvis.save_html(vis_data, out_html)
        print(f"可视化结果已保存: {out_html}")
    except Exception as e:
        print("pyLDAvis 可视化出错:", e)


if __name__=="__main__":
    main()
