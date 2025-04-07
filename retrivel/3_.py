import json
import os
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import Counter
import math
import random

# 语义模型
tokenizer = AutoTokenizer.from_pretrained("../unixcoder-base")
model = AutoModel.from_pretrained("../unixcoder-base")


def bm25_retrieval1(query, chunked_lists):
    bm25 = BM25Okapi([chunk["truncated_chunk"].split() for chunk in chunked_lists])
    scores = bm25.get_scores(query.split())
    return scores

def bm25_retrieval(query, chunked_lists):
    tokenized_corpus = [tokenizer.tokenize(chunk["truncated_chunk"]) for chunk in chunked_lists]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenizer.tokenize(query))
    return scores

def bm25_retrieval2(query, chunked_lists, weight_type='sqrt'):
    # 分割查询并生成线性权重
    # query_terms = query.split()
    query_terms = tokenizer.tokenize(query)

    if weight_type == 'line':
        weights = {word: i+1 for i, word in enumerate(query_terms)}  # 线性加权，从1递增到词的数量
    elif weight_type == '2^x':
        weights = {word: 2 ** (i + 1) for i, word in enumerate(query_terms)}  # 指数加权，使用2的幂次作为权重
    elif weight_type == 'e^x':
        weights = {word: math.exp(i + 1) for i, word in enumerate(query_terms)}  # 使用e的指数作为权重
    elif weight_type == 'log':
        weights = {word: math.log(i + 2) for i, word in enumerate(query_terms)}  # 使用对数加权
    elif weight_type == 'sqrt':
        weights = {word: (i + 1) ** 5 for i, word in enumerate(query_terms)}  # 使用平方根加权
    elif weight_type == 'sigmoid':
        weights = {word: 1 / (1 + math.exp(-i - 1)) for i, word in enumerate(query_terms)}  # 使用Sigmoid加权
    else:
        raise ValueError("Unknown weight type")

    # 准备文档
    # tokenized_corpus = [chunk["truncated_chunk"].split() for chunk in chunked_lists]
    tokenized_corpus = [tokenizer.tokenize(chunk["truncated_chunk"]) for chunk in chunked_lists]
    bm25 = BM25Okapi(tokenized_corpus)

    # 手动计算加权BM25得分
    def get_weighted_bm25_scores(query, weights, bm25, corpus):
        doc_scores = []
        for document in corpus:
            doc_dict = Counter(document)
            doc_len = len(document)
            score = 0
            for word in query:
                if word in bm25.idf and word in doc_dict:
                    term_freq = doc_dict[word]
                    idf = bm25.idf[word]
                    term_weight = weights.get(word, 1)
                    # 应用加权的词频计算公式
                    weighted_tf = (term_freq * (bm25.k1 + 1) * term_weight) / (term_freq + bm25.k1 * (1 - bm25.b + bm25.b * (doc_len / bm25.avgdl)))
                    score += idf * weighted_tf
            doc_scores.append(score)
        return doc_scores

    # 计算加权得分
    scores = get_weighted_bm25_scores(query_terms, weights, bm25, tokenized_corpus)
    return scores


def bert_retrieval(query, chunked_lists, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # 半精度推理
    with autocast():
        with torch.no_grad():
            query_embedding = model(**query_tokens).last_hidden_state.mean(1)

    # 准备批量处理chunks
    chunk_texts = [chunk["truncated_chunk"] for chunk in chunked_lists]
    chunk_encodings = tokenizer(chunk_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    chunk_dataset = TensorDataset(chunk_encodings['input_ids'].to(device), chunk_encodings['attention_mask'].to(device))
    chunk_dataloader = DataLoader(chunk_dataset, sampler=SequentialSampler(chunk_dataset), batch_size=batch_size)

    scores = []
    model.eval()  # 在推理时调用 model.eval()
    with autocast():
        with torch.no_grad():
            for batch in chunk_dataloader:
                input_ids, attention_mask = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                chunk_embeddings = outputs.last_hidden_state.mean(1)
                # 计算相似度时需要将数据移回CPU
                batch_scores = cosine_similarity(query_embedding.cpu().numpy(), chunk_embeddings.cpu().numpy())
                scores.extend(batch_scores.flatten())

    # 将模型恢复到CPU
    model.to("cpu")

    return scores


#读取源文件，并额外提取其中的 prompt 和 repository 字段作为查询 和用于匹配同名仓库
def read_source_file(source_file_path):
    prompt_list = []
    repositories = []
    filename_list = []
    source_data = []  # 新增，用于存储完整的源数据行

    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            data = json.loads(line.strip())

            # 这里不再分别提取各个字段，而是直接将整个data字典添加到source_data列表中
            source_data.append(data)

            # 继续提取需要的字段，以便于后续处理
            query = data['prompt']
            repository = data['metadata']['repository']
            filename = data['metadata']['file']

            prompt_list.append(query)
            repositories.append(repository)
            filename_list.append(filename)

    # 返回包括新的source_data列表在内的四个值
    return prompt_list, repositories, filename_list, source_data



#所有行的chunked_list合并在一起当做被检索对象，并且排除源文件中的文件名与被检索文件相同的情况。
def retrieve_similar_chunks(source_queries, source_repositories, source_filename_list, retrieval_folder, N, retrieval_model):
    all_similar_chunks = []  # 这将是一个列表的列表，其中每个子列表包含对应一个查询的所有相似文本块

    for query, source_repository, source_filename in tqdm(zip(source_queries, source_repositories, source_filename_list), total=len(source_queries), desc="Retrieving similar chunks"):
        matching_filename = os.path.join(retrieval_folder, source_repository.replace('/', '_') + '.jsonl')
        if not os.path.exists(matching_filename):
            print("特别注意，打印这个说明出错了，必须能匹配到待检索文件夹，否则数据集构造会出错！"
                  "说明对于给定的上下文代码，没有找到其对应的仓库，也就无法进行跨文件检索。")
            print("error repository_filename: ", matching_filename)

            continue

        chunked_lists = []
        with open(matching_filename, 'r') as retrieval_file:
            for line in retrieval_file:
                data = json.loads(line.strip())
                chunked_file_name = data['filename']
                if chunked_file_name == source_filename:  # 排除同一文件
                    continue

                for chunk in data['chunked_list']:
                    truncated_chunk = chunk

                    chunked_lists.append({
                        "original_chunk": chunk,
                        "truncated_chunk": truncated_chunk,
                        "filename": chunked_file_name
                    })


        query_tokens = tokenizer.tokenize(query)[-512:]
        truncated_query = tokenizer.convert_tokens_to_string(query_tokens)

        # truncated_query = query

        if retrieval_model == 'BM25':
            scores = bm25_retrieval(truncated_query, chunked_lists)
        elif retrieval_model == 'Semantic_Model':
            scores = bert_retrieval(truncated_query, chunked_lists)
        else:
            raise ValueError("Unsupported retrieval model. Choose either 'BM25' or 'BERT'.")

        max_score_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]

        # 为当前查询创建一个新的相似文本块列表
        query_similar_chunks = []
        for idx in max_score_indices:
            similar_chunk = chunked_lists[idx]
            query_similar_chunks.append({
                "retrieved_chunk": similar_chunk["original_chunk"],
                "filename": similar_chunk["filename"],
                "score": float(scores[idx])
            })

        # 将当前查询的相似文本块列表添加到最终结果列表中
        all_similar_chunks.append(query_similar_chunks)

    return all_similar_chunks



def integrate_and_write_output(source_data, similar_chunks, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for data, chunks in zip(source_data, similar_chunks):

            text = "# Here are some relevant code fragments from other files of the repo:\n"
            for chunk in chunks:
                text += "# the below code fragment can be found in:\n"
                text += "# " + chunk["filename"] + "\n"
                text += "\n".join("# " + line for line in chunk["retrieved_chunk"].split('\n')) + "\n"

            # 构造crossfile_context部分
            crossfile_context = {
                "text": text,  # 如果需要，此处可以填充具体的文本
                "list": chunks  # 直接使用对应的相似文本块列表
            }

            # 更新源数据中的crossfile_context部分
            new_data = data.copy()  # 复制原始数据以避免修改原始数据  # 这保留原始数据的其他所有部分
            new_data["crossfile_context"] = crossfile_context

            # 写入更新后的数据到输出文件
            output_file.write(json.dumps(new_data) + '\n')




def main():
    # 定义路径（假设路径和文件名已正确设置）
    source_file_path = "../data/typescript/line_completion_rg1_bm25.jsonl"  #修改的参数1  被替换文件
    # retrieval_folder = "./chunk_folder/typescript"#修改的参数2         #从哪里检索代码块，实际上就是第二步分块后的文件，所保存的文件夹
    retrieval_folder = "../slide_chunk/typescript_slide_chunked_1010"  #
    output_file_path = "../data/typescript/test_origin.jsonl" #修改的参数3， 最后一个参数  #保存的格式和原作者的data格式会一致
    retrieval_model = "BM25" #BM25 Semantic_Model #可选的修改参数  #检索模型
    N = 100  # Top N similar chunks to retrieve

    # 步骤1: 读取源文件
    source_queries, source_repositories, source_filename_list, source_data = read_source_file(source_file_path)

    # 步骤2: 检索相似文本块
    similar_chunks = retrieve_similar_chunks(source_queries, source_repositories, source_filename_list, retrieval_folder, N, retrieval_model)
    print(len(similar_chunks))

    # 步骤3: 整合数据和相似文本块，并写入输出文件
    integrate_and_write_output(source_data, similar_chunks, output_file_path)

if __name__ == "__main__":
    main()
