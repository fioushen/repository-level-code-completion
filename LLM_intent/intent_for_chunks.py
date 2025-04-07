import os
import json
from openai import OpenAI

# 初始化 OpenAI 客户端
# client = OpenAI(api_key="sk-cf4bff9bb77b41468128805bc41db79e", base_url="https://api.deepseek.com/")

# 源文件夹和目标文件夹路径
src_directory = './typescript_chunks_test'
dest_directory = './typescript_chunks_test_out'
os.makedirs(dest_directory, exist_ok=True)

# def summarize_function(code_chunk):
#     print("code_chunk")
#     print(code_chunk)
#     print("------------------------")
#
#     """通过远程 API 总结代码块的功能"""
#     messages = [
#         {"role": "system", "content": "Summarize the intent of the code in one sentence"},
#         {"role": "user", "content": code_chunk},
#     ]
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=messages,
#         temperature=0.5
#     )
#     # 获取 API 返回的内容
#     print("llm response")
#     print(response.choices[0].message.content) #LLM_response_content = response.choices[0].message.content
#     print("==========================")
#     return response.choices[0].message.content
import re
def extract_comments_for_python(code):
    # 正则表达式匹配单行注释和多行注释
    single_line_comment_pattern = r'#.*'
    multi_line_comment_pattern = r'\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\"'

    # 提取单行注释
    single_line_comments = re.findall(single_line_comment_pattern, code)

    # 提取多行注释
    multi_line_comments = re.findall(multi_line_comment_pattern, code, re.DOTALL)
    multi_line_comments = [comment[0] if comment[0] else comment[1] for comment in multi_line_comments]

    # 合并所有注释
    all_comments = single_line_comments + multi_line_comments

    # 用换行符分隔每个注释
    return '\n'.join(all_comments)

def process_files():
    """处理文件夹中的所有 jsonl 文件"""
    for filename in os.listdir(src_directory):
        print(filename)
        if filename.endswith('.jsonl'):
            with open(os.path.join(src_directory, filename), 'r') as file, \
                 open(os.path.join(dest_directory, filename), 'w') as outfile:

                for line in file:
                    data = json.loads(line)
                    function_list = [chunk[0:500000] for chunk in data['chunked_list']]
                    # function_list = [extract_comments_for_python(chunk) for chunk in data['chunked_list']]
                    data['function_list'] = function_list
                    json.dump(data, outfile)
                    outfile.write('\n')

if __name__ == '__main__':
    process_files()


'''
帮我写代码，我有一个文件夹，里面有若干个jsonl文件，每个jsonl文件的每一行格式是：
{"filename": "citrusdb/db/sqlite/__init__.py", "chunked_list": [块1，块2]}。

我的要求是：
读取这个文件夹中所有的jsonl文件，对每个jsonl文件进行如下的操作：
把每一行的所有块取出来，对于每个块，我们通过连接远程LLM的api，总结出他的功能，然后将块和其对应的功能，保存到新的文件中。

文件名和原来的jsonl文件名字一样，只是文件夹不一样，每一行的格式是
{"filename": "citrusdb/db/sqlite/__init__.py", "chunked_list": [块1，块2]，"function_list":[块1的功能，块2的功能] }。
filename和以前也一致，chunked_list和以前也一致，只是多了function_list

'''