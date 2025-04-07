import os
import json

#接收一个文件路径和一个分块大小（默认为10行）。它读取文件，去除空行，然后将代码按指定的行数分割成小块。
# 每个小块由指定数量的代码行组成，这些行用制表符（tab）连接。最后，这些代码块以列表的形式返回。
def chunk_code(file_path, chunk_size=10):
    with open(file_path, 'r', encoding='utf-8',errors='ignore') as f:  #errors='ignore' 在读csharp时，有的字节解码错误，先忽略错误的字节
        lines = [line for line in f.readlines() if line.strip()]  # 过滤掉没有内容的行
        chunks = ['	'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)] #到底是join空格，还是tab？目前先用tab。
    return chunks



#遍历给定语言的所有代码文件，对每个文件使用chunk_code函数进行分块，然后将结果（文件路径和对应的代码块列表）保存到一个JSONL文件中。
# 文件名是仓库名加上.jsonl后缀，每行是一个JSON对象，表示一个文件及其分块。
def save_chunks_to_jsonl(repo_path, language, chunk_folder, chunk_size):
    def get_files_in_folder(folder_path, extension):
        files = []
        for root, _, filenames in os.walk(folder_path, followlinks=True):  # 跟随符号链接
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return files

    files = get_files_in_folder(repo_path, '.' + language)
    repo_name = os.path.basename(os.path.normpath(repo_path))
    output_path = os.path.join(chunk_folder, f'{repo_name}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file in files:
            try:
                chunks = chunk_code(file, chunk_size)
                relative_path = os.path.relpath(file, repo_path)
                filename = relative_path.replace(os.sep, '/')
                json.dump({'filename': filename, 'chunked_list': chunks}, outfile, ensure_ascii=False)
                outfile.write('\n')
            except FileNotFoundError as e:
                print(f"File not found: {file}. Skipping...") #由于符号链接问题，文件可能只有链接，但实际不存在，所以就跳过

#遍历指定文件夹下的所有仓库，对每个仓库调用save_chunks_to_jsonl函数，根据指定的语言和分块大小处理并保存分块。
def process_repositories(repo_folder, language, chunk_folder, chunk_size):
    for repo_name in os.listdir(repo_folder):
        repo_path = os.path.join(repo_folder, repo_name)
        if os.path.isdir(repo_path):
            save_chunks_to_jsonl(repo_path, language, chunk_folder, chunk_size)



#一次只处理一个语言
#####################手动修改这里#############
repo_folder = './sorted_repositories/typescript'  #要修改的参数1     #分类后的仓库的文件夹
language = 'ts'  # ['java', 'cs', 'ts', 'py'] #要修改的参数2 #语言的后缀
chunk_folder = './chunk_folder/typescript' #要修改的参数3 最后一个参数  #分块后的文件，保存的文件夹
chunk_size = 10  #可选的参数 #分块大小

# Create the chunk folder if it doesn't exist
if not os.path.exists(chunk_folder):
    os.makedirs(chunk_folder)

# Process repositories
process_repositories(repo_folder, language, chunk_folder, chunk_size)

print("由于符号链接问题，文件可能只有链接，但实际不存在，所以就跳过,没多大问题")
#
# #####################################################################


#一次性处理所有语言

# chunk_size = 10  # Specify the chunk size you desire #可选的参数
# repo_folder_list = ['java', 'csharp', 'typescript', 'python']
# language_list = ['java', 'cs', 'ts', 'py']
#
# for rf, language in zip(repo_folder_list, language_list):
#     repo_folder = './sorted_repositories/' + rf    # 指定的仓库文件夹路径
#     chunk_folder = './chunk_folder/' + rf  # 分块结果的保存路径
#     # Create the chunk folder if it doesn't exist
#     if not os.path.exists(chunk_folder):
#         os.makedirs(chunk_folder)
#
#     # Process repositories
#     process_repositories(repo_folder, language, chunk_folder, chunk_size)

