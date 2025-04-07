import json
import os
import shutil




def process_jsonl_and_copy_folders(jsonl_file_path, source_folder_root, target_root_folder):
    unique_repositories = set()

    # 从JSONL文件中读取和处理数据
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            # 提取repository字段
            repository = data['metadata']['repository']
            unique_repositories.add(repository)

    # 遍历唯一的仓库名称，并复制对应的子文件夹
    for repository in unique_repositories:
        # 构建源文件夹和目标文件夹的路径
        source_folder = os.path.join(source_folder_root, repository)
        target_folder = os.path.join(target_root_folder, repository)

        # 检查源文件夹是否存在
        if os.path.exists(source_folder):
            # 复制整个仓库文件夹
            shutil.copytree(source_folder, target_folder, dirs_exist_ok=True, symlinks=True)
            # 移动整个仓库文件夹
            # shutil.move(source_folder, target_folder)
            # print(f"Copied repository {repository} to {target_folder}")
        else:
            print(f"Repository folder not found: {source_folder}")




if __name__ == '__main__':

    for lang in ["python", "java", "csharp", "typescript"]:
        # 输入的JSONL文件路径
        jsonl_file_path = '../data/' + lang + '/line_completion.jsonl'
        # 源文件夹路径，即包含所有仓库子文件夹的根目录
        source_folder_root = './crosscodeeval_rawdata_v1.1/crosscodeeval_rawdata'
        # 目标根文件夹路径，即要复制到的目录
        target_root_folder = './sorted_repositories/' + lang

        # 确保目标根文件夹存在
        os.makedirs(target_root_folder, exist_ok=True)

        # 执行脚本
        process_jsonl_and_copy_folders(jsonl_file_path, source_folder_root, target_root_folder)