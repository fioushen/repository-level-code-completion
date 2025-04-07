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
# tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
# model = AutoModel.from_pretrained("microsoft/unixcoder-base")

tokenizer = AutoTokenizer.from_pretrained("../unixcoder-base")
model = AutoModel.from_pretrained("../unixcoder-base")

def bm25_retrieval(query, chunked_lists):
    tokenized_corpus = [tokenizer.tokenize(chunk["truncated_chunk"]) for chunk in chunked_lists]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenizer.tokenize(query))
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


# def calculate_intent_similarity(query_intent, chunk_intents, model, tokenizer, device):
#     query_intent_encoding = tokenizer(query_intent, return_tensors="pt", padding=True, truncation=True,
#                                       max_length=512).to(device)
#     chunk_intent_encodings = tokenizer(chunk_intents, padding=True, truncation=True, max_length=512,
#                                        return_tensors="pt").to(device)
#
#     with torch.no_grad():
#         query_intent_embedding = model(**query_intent_encoding).last_hidden_state.mean(1)
#         chunk_intent_embeddings = model(**chunk_intent_encodings).last_hidden_state.mean(1)
#         similarities = cosine_similarity(query_intent_embedding.cpu().numpy(), chunk_intent_embeddings.cpu().numpy())
#
#     return similarities.flatten()


def calculate_intent_similarity(query_intent, chunk_intents, model, tokenizer, device, batch_size=64):
    # Tokenize and encode the query intent
    query_intent_encoding = tokenizer(query_intent, return_tensors="pt", padding=True, truncation=True,
                                      max_length=512).to(device)

    # Compute the query intent embedding
    with torch.no_grad():
        query_intent_embedding = model(**query_intent_encoding).last_hidden_state.mean(1)

    # Prepare to collect all chunk intent embeddings
    all_chunk_embeddings = []

    # Process chunk intents in batches
    for i in range(0, len(chunk_intents), batch_size):
        batch_intents = chunk_intents[i:i + batch_size]
        batch_encodings = tokenizer(batch_intents, padding=True, truncation=True, max_length=512,
                                    return_tensors="pt").to(device)

        with torch.no_grad():
            batch_embeddings = model(**batch_encodings).last_hidden_state.mean(1)
            all_chunk_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings on axis 0 (stacking vertically)
    all_chunk_embeddings = torch.cat(all_chunk_embeddings, dim=0)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_intent_embedding.cpu().numpy(), all_chunk_embeddings.cpu().numpy())

    return similarities.flatten()




#读取源文件，并额外提取其中的 prompt 和 repository 字段作为查询 和用于匹配同名仓库
def read_source_file(source_file_path):
    prompt_list = []
    repositories = []
    filename_list = []
    source_data = []  # 新增，用于存储完整的源数据行
    prompt_function_list = []

    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            data = json.loads(line.strip())

            # 这里不再分别提取各个字段，而是直接将整个data字典添加到source_data列表中
            source_data.append(data)

            # 继续提取需要的字段，以便于后续处理
            query = data['prompt']
            prompt_function = data['prompt_function']
            repository = data['metadata']['repository']
            filename = data['metadata']['file']

            prompt_list.append(query)
            repositories.append(repository)
            filename_list.append(filename)
            prompt_function_list.append(prompt_function)

    # 返回包括新的source_data列表在内的四个值
    return prompt_list, prompt_function_list, repositories, filename_list, source_data



#所有行的chunked_list合并在一起当做被检索对象，并且排除源文件中的文件名与被检索文件相同的情况。
def retrieve_similar_chunks(source_queries, prompt_function_list, source_repositories, source_filename_list,
                            retrieval_folder, N, retrieval_model, p):
    all_similar_chunks = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for query, prompt_function, source_repository, source_filename in tqdm(
            zip(source_queries, prompt_function_list, source_repositories, source_filename_list),
            total=len(source_queries), desc="Retrieving similar chunks"):
        matching_filename = os.path.join(retrieval_folder, source_repository.replace('/', '_') + '.jsonl')
        if not os.path.exists(matching_filename):
            print("Error: Matching file not found for repository:", matching_filename)
            continue

        chunked_lists = []
        with open(matching_filename, 'r') as retrieval_file:
            for line in retrieval_file:
                data = json.loads(line.strip())
                if data['filename'] == source_filename:
                    continue

                for chunk, chunk_intent in zip(data['chunked_list'], data['function_list']):
                    chunked_lists.append({
                        "original_chunk": chunk,
                        "truncated_chunk": chunk,
                        "filename": data['filename'],
                        "chunk_intent": chunk_intent
                    })

        # truncated_query_lines = [line for line in query.split('\n') if line.strip()]
        #truncated_query_lines = [line for line in query.split('\n') if line.strip()][-5:]
        truncated_query_lines = query.split('\n')[-20:]
        # truncated_query_lines = query.split('\n')[-5:]
        truncated_query = "\n".join(truncated_query_lines)
        query = truncated_query


        if retrieval_model == 'BM25':
            scores = bm25_retrieval(query, chunked_lists)
        elif retrieval_model == 'Semantic_Model':
            scores = bert_retrieval(query, chunked_lists)
        else:
            raise ValueError("Unsupported retrieval model. Choose either 'BM25' or 'BERT'.")



        # print(query)
        # print("---")
        # print(prompt_function)
        # print("==================")
        # print()


        chunk_intents = [chunk['chunk_intent'] for chunk in chunked_lists]
        intent_similarities = calculate_intent_similarity(prompt_function, chunk_intents, model, tokenizer, device)

        # 相当于可以设置调和参数，将意图相似度，和待补全代码与块的文本相似度，结合起来

        combined_scores = [p * sim + (1-p) * score for sim, score in zip(intent_similarities, scores)]
        max_score_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:N]

        query_similar_chunks = []
        for idx in max_score_indices:
            similar_chunk = chunked_lists[idx]
            query_similar_chunks.append({
                "retrieved_chunk": similar_chunk["original_chunk"],
                "filename": similar_chunk["filename"],
                "score": float(combined_scores[idx])
            })

        all_similar_chunks.append(query_similar_chunks)

    model.to("cpu")
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
    source_file_path = "./python_data_out/line_completion_rg1_bm25.jsonl"  #修改的参数1  原作者制作的data目录
    retrieval_folder = "./python_chunks_test_out"
    retrieval_model = "BM25" #BM25 Semantic_Model #可选的修改参数  #检索模型
    N = 100  # Top N similar chunks to retrieve

    for p in [1]:
        output_file_path = "../data/python/test_intent_weight_"+str(p)+".jsonl"  # 修改的参数3， 最后一个参数  #保存的格式和原作者的data格式会一致
        print(output_file_path)

        # 步骤1: 读取源文件
        source_queries, prompt_function_list, source_repositories, source_filename_list, source_data = read_source_file(source_file_path)

        # 步骤2: 检索相似文本块
        similar_chunks = retrieve_similar_chunks(source_queries, prompt_function_list, source_repositories,
                                                 source_filename_list, retrieval_folder, N, retrieval_model, p)
        print(len(similar_chunks))

        # 步骤3: 整合数据和相似文本块，并写入输出文件
        integrate_and_write_output(source_data, similar_chunks, output_file_path)

if __name__ == "__main__":
    main()
