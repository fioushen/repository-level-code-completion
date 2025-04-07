import ast
import os
import json


def get_code_blocks(source_code):
    try:
        parsed_code = ast.parse(source_code)
    except SyntaxError:
        return None  # 返回None，表示解析失败
    code_blocks = []
    for node in ast.walk(parsed_code):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.Try, ast.With)):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            code_blocks.append((start_line, end_line))
    return code_blocks


def adjust_position(position, blocks, direction):
    for block_start, block_end in blocks:
        if block_start <= position < block_end:
            return block_start if direction == 'backward' else block_end
    return position


def chunk_code(source_code, window_size, slide_size):
    code_blocks = get_code_blocks(source_code)

    if code_blocks is None:  # 解析失败，使用滑动窗口方法分割代码
        lines = source_code.split('\n')
        return ['\n'.join(lines[i:i + window_size]) for i in range(0, len(lines), slide_size)]

    chunks = []
    current_position = 0
    lines = source_code.split('\n')

    while current_position < len(lines):
        end_position = min(current_position + window_size, len(lines))
        end_position = adjust_position(end_position, code_blocks, 'forward')

        chunk = '\n'.join(lines[current_position:end_position])
        chunks.append(chunk)

        next_position = current_position + slide_size
        for block_start, block_end in code_blocks:
            if block_start < next_position < block_end:
                next_position = block_end
                break
        current_position = next_position

    return chunks


def process_repository(repo_path, output_dir, window_size=10, slide_size=8):
    repo_name = os.path.basename(repo_path)
    output_file_path = os.path.join(output_dir, f"{repo_name}.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()
                    except FileNotFoundError:
                        continue  # 文件未找到，跳过当前文件
                    chunks = chunk_code(source_code, window_size, slide_size)
                    relative_path = os.path.relpath(file_path, repo_path)
                    output_line = {"filename": relative_path, "chunked_list": chunks}
                    output_file.write(json.dumps(output_line) + '\n')

def main():
    repos_path = "../../retrivel/sorted_repositories/python"
    output_dir = "../python_slide_chunked"
    os.makedirs(output_dir, exist_ok=True)
    for repo in os.listdir(repos_path):
        repo_path = os.path.join(repos_path, repo)
        if os.path.isdir(repo_path):
            process_repository(repo_path, output_dir, window_size=10, slide_size=8)

if __name__ == '__main__':
    main()
