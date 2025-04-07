import re

def get_code_blocks(source_code):
    patterns = [
        r'public\s+class\s+\w+',          # Class definition
        r'public\s+(static\s+)?void\s+\w+', # Function definition (void)
        r'public\s+(static\s+)?\w+\s+\w+',  # Function definition (non-void)
        r'if\s*\(',                       # If statement
        r'for\s*\(',                      # For loop
        r'while\s*\(',                    # While loop
        r'try\s*{',                       # Try block
        r'with\s*\(',                     # With statement
    ]
    pattern = '|'.join(patterns)
    code_blocks = []
    lines = source_code.split('\n')
    block_start = None
    for i, line in enumerate(lines):
        if re.search(pattern, line.strip()):
            if block_start is not None:
                code_blocks.append((block_start, i))
            block_start = i
    if block_start is not None:
        code_blocks.append((block_start, len(lines)))
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
source_code = """import java.util.ArrayList;
import java.util.List;

public class Example {
    public static void main(String[] args) {
        // Create a list of integers
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            numbers.add(i);
        }

        // Print the list
        System.out.println("Numbers: " + numbers);

        // Calculate the sum of the list
        int sum = 0;
        for (int number : numbers) {
            sum += number;
        }

        // Print the sum
        System.out.println("Sum: " + sum);

        // Find the maximum number in the list
        int max = numbers.get(0);
        for (int number : numbers) {
            if (number > max) {
                max = number;
            }
        }

        // Print the maximum number
        System.out.println("Max: " + max);

        // Find the minimum number in the list
        int min = numbers.get(0);
        for (int number : numbers) {
            if (number < min) {
                min = number;
            }
        }

        // Print the minimum number
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
        System.out.println("Min: " + min);
    }
}
"""

chunks = chunk_code(source_code, 5, 3)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
