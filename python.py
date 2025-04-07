import ast


def get_code_blocks(source_code):
    parsed_code = ast.parse(source_code)
    code_blocks = []
    for node in ast.walk(parsed_code):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef,  ast.If, ast.For, ast.Try, ast.With)): #ast.ClassDef,
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





# 示例代码
source_code = """# Function
def greet(name):
    print(f"Hello, {name}!")

# Class
class Dog:
    def __init__(self, name):
        self.name = name
    
    def bark1(self):
        print(f"{self.name} says woof!")
        
    def bark2(self):
        print(f"{self.name} says woof!")
    
    def bark3(self):
        print(f"{self.name} says woof!")

# If statement
x = 10
if x > 5:
    print("x is greater than 5")

# For loop
for i in range(3):
    print(f"Count: {i}")

# While loop
num = 3
while num > 0:
    print(f"Num: {num}")
    num -= 1

# Try-except block
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Division by zero!")

# With statement
with open('example.txt', 'w') as f:
    f.write('Hello, world!')
"""

chunks = chunk_code(source_code, 5, 3)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
#这是python的分块程序，gpt生成的