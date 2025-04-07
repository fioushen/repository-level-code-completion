import re


# 获取不能被截断的基本块
def get_code_blocks(source_code):
    patterns = [
        r'\bclass\s+\w+',  # Class definition
        r'\bstruct\s+\w+',  # Struct definition
        r'\benum\s+\w+',  # Enum definition
        r'\binterface\s+\w+',  # Interface definition
        r'\bdelegate\s+\w+',  # Delegate definition
        r'\bpublic\s+(static\s+)?\w+\s+\w+',  # Function definition
        r'\bprivate\s+(static\s+)?\w+\s+\w+',  # Private function definition
        r'\bprotected\s+(static\s+)?\w+\s+\w+',  # Protected function definition
        r'\bif\s*\(',  # If statement
        r'\bfor\s*\(',  # For loop
        r'\bwhile\s*\(',  # While loop
        r'\bforeach\s*\(',  # Foreach loop
        r'\btry\s*{',  # Try block
        r'\bcatch\s*\(',  # Catch block
        r'\bfinally\s*{',  # Finally block
        r'\busing\s*\(',  # Using statement
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

# 调整截断位置，确保块完整
def adjust_position(position, blocks, direction):
    for block_start, block_end in blocks:
        if block_start <= position < block_end:
            return block_start if direction == 'backward' else block_end
    return position

# 分块代码
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

# 示例C#代码
source_code = """
using System;

namespace ExampleNamespace
{
    // 枚举
    public enum DaysOfWeek
    {
        Monday,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday
    }

    // 委托
    public delegate void MessageHandler(string message);

    // 接口
    public interface IExampleInterface
    {
        void DisplayMessage(string message);
    }

    // 结构体
    public struct Point
    {
        public int X;
        public int Y;
    }

    // 类
    public class ExampleClass : IExampleInterface
    {
        private int number;

        // 构造函数
        public ExampleClass(int num)
        {
            number = num;
        }

        // 方法
        public void DisplayNumber()
        {
            Console.WriteLine("Number: " + number);
        }

        // 接口方法实现
        public void DisplayMessage(string message)
        {
            Console.WriteLine("Message: " + message);
        }

        // 条件语句
        public void CheckNumber(int num)
        {
            if (num > 0)
                Console.WriteLine("Number is positive");
            else if (num < 0)
                Console.WriteLine("Number is negative");
            else
                Console.WriteLine("Number is zero");
        }

        // 循环
        public void PrintNumbers(int start, int end)
        {
            for (int i = start; i <= end; i++)
            {
                Console.Write(i + " ");
            }
            Console.WriteLine();
        }

        // 析构函数
        ~ExampleClass()
        {
            Console.WriteLine("Object is being destroyed");
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 实例化一个类对象
            ExampleClass exampleObj = new ExampleClass(10);
            exampleObj.DisplayNumber();
            exampleObj.DisplayMessage("Hello World");
            exampleObj.CheckNumber(5);
            exampleObj.PrintNumbers(1, 5);

            // 使用枚举
            DaysOfWeek today = DaysOfWeek.Monday;
            Console.WriteLine("Today is " + today);

            // 使用委托
            MessageHandler handler = delegate (string message)
            {
                Console.WriteLine("Delegate Message: " + message);
            };
            handler("This is a delegate message");

            // 使用异常处理块
            try
            {
                int result = 10 / 0; // Divide by zero exception
            }
            catch (DivideByZeroException ex)
            {
                Console.WriteLine("Exception Caught: " + ex.Message);
            }
        }
    }
}

"""

# 设置窗口大小和滑动大小
window_size = 10
slide_size = 8

# 获取分块结果
chunks = chunk_code(source_code, window_size, slide_size)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
