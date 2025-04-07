import re



def get_code_blocks(source_code):
    patterns = [
        r'\bclass\s+\w+',  # Class definition
        r'\binterface\s+\w+',  # Interface definition
        r'\benum\s+\w+',  # Enum definition
        r'\bmodule\s+\w+',  # Module definition
        r'\bfunction\s+\w+',  # Function definition
        r'\bif\s*\(',  # If statement
        r'\bfor\s*\(',  # For loop
        r'\bwhile\s*\(',  # While loop
        r'\btry\s*{',  # Try block
        r'\bcatch\s*\(',  # Catch block
        r'\bfinally\s*{',  # Finally block
        r'\bwith\s*\(',  # With statement
        r'\btype\s+\w+',  # Type alias
        r'\bgeneric\s*\<',  # Generic definition
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

# 示例C#代码
source_code = """
import * as ts from '1';
import * as fs from '2';
import * as path from '3';
import * as ts from '5';
import * as fs from '6';
import * as path from '7';

import * as ts from '8';
import * as fs from '9';
import * as path from '10 ';

import * as ts from '11';
import * as fs from '12';
import * as path from '13';


// 解析 TypeScript 代码并生成 AST
function parseTypescript(code: string): ts.SourceFile {
    return ts.createSourceFile('temp.ts', code, ts.ScriptTarget.Latest);
}

// 判断节点是否为基本块（函数、类、接口、枚举、模块、泛型）
function isBasicBlock(node: ts.Node): boolean {
    return ts.isFunctionDeclaration(node) ||
        ts.isClassDeclaration(node) ||
        ts.isInterfaceDeclaration(node) ||
        ts.isEnumDeclaration(node) ||
        ts.isModuleDeclaration(node) ||
        ts.isTypeAliasDeclaration(node)||
        ts.isVariableStatement(node); // 变量声明
}

// 判断节点类型
function getNodeType(node: ts.Node): string {
    if (ts.isFunctionDeclaration(node)) return 'Function';
    if (ts.isClassDeclaration(node)) return 'Class';
    if (ts.isInterfaceDeclaration(node)) return 'Interface';
    if (ts.isEnumDeclaration(node)) return 'Enum';
    if (ts.isModuleDeclaration(node)) return 'Module';
    if (ts.isTypeAliasDeclaration(node)) return 'TypeAlias';
    if (ts.isVariableStatement(node)) return 'Variable'; // 变量声明
    return 'Unknown';
}

// 创建打印机
const printer = ts.createPrinter();

// 遍历 AST，提取基本块，并按类型分类保存到列表中
function extractBlocks(sourceFile: ts.SourceFile, threshold: number): { [key: string]: ts.Node[] } {
    const extractedBlocks: { [key: string]: ts.Node[] } = {
        'Function': [],
        'Class': [],
        'Interface': [],
        'Enum': [],
        'Module': [],
        'TypeAlias': [],
        'InterBlock': [], // 添加用于保存块与块之间连续代码的列表
        'Variable': [],
        'Unknown': []
    };

    let interBlockStatements: ts.Statement[] = [];
    let currentBlock: ts.Node | null = null;

    function visit(node: ts.Node) {
        if (isBasicBlock(node)) {
            if (interBlockStatements.length > 0 && currentBlock !== null) {
                extractedBlocks['InterBlock'].push(ts.factory.createBlock(interBlockStatements, true));
                interBlockStatements = [];
            }
            currentBlock = node;

            const nodeType = getNodeType(node);
            if (nodeType === 'Class' && ts.isClassDeclaration(node)) {
                const classNode = node as ts.ClassDeclaration;
                if (classNode.members.length > threshold) {
                    classNode.members.forEach(member => extractedBlocks[nodeType].push(member));
                } else {
                    extractedBlocks[nodeType].push(node);
                }
            } else if (nodeType === 'Module' && ts.isModuleDeclaration(node)) {
                const moduleNode = node as ts.ModuleDeclaration;
                if (moduleNode.body && ts.isModuleBlock(moduleNode.body)) {
                    const statements = moduleNode.body.statements;
                    if (statements.length > threshold) {
                        statements.forEach(statement => extractedBlocks[nodeType].push(statement));
                    } else {
                        extractedBlocks[nodeType].push(node);
                    }
                } else if (moduleNode.body && ts.isModuleDeclaration(moduleNode.body)) {
                    visit(moduleNode.body);
                }
            } else {
                extractedBlocks[nodeType].push(node);
            }
        } else {
            if (currentBlock !== null && ts.isStatement(node)) {
                interBlockStatements.push(node);
            }
            ts.forEachChild(node, visit);
        }
    }

    visit(sourceFile);

    // 如果最后还有剩余的 interBlockStatements，也要加入到 InterBlock 类型中
    if (interBlockStatements.length > 0) {
        extractedBlocks['InterBlock'].push(ts.factory.createBlock(interBlockStatements, true));
    }

    return extractedBlocks;
}

// 处理一个 TypeScript 文件，提取并返回基本块
function processFile(filePath: string, threshold: number): { [key: string]: string[] } {
    const code = fs.readFileSync(filePath, 'utf8');
    const sourceFile = parseTypescript(code);
    const extractedBlocks = extractBlocks(sourceFile, threshold);

    // 将提取的基本块转换为字符串
    const blockStrings: { [key: string]: string[] } = {};
    Object.keys(extractedBlocks).forEach(type => {
        blockStrings[type] = extractedBlocks[type].map(block => printer.printNode(ts.EmitHint.Unspecified, block, sourceFile));
    });

    return blockStrings;
}

// 遍历仓库目录，处理每个 TypeScript 文件，并将结果保存到 JSONL 文件中
function processRepository(repoPath: string, outputDir: string, threshold: number): void {
    const repoName = path.basename(repoPath);
    const jsonlFilePath = path.join(outputDir, `${repoName}.jsonl`);
    const jsonlFile = fs.createWriteStream(jsonlFilePath);

    function processDirectory(directoryPath: string): void {
        fs.readdirSync(directoryPath, { withFileTypes: true }).forEach(dirent => {
            const fullPath = path.join(directoryPath, dirent.name);
            if (dirent.isDirectory()) {
                processDirectory(fullPath);
            } else if (dirent.isFile() && path.extname(dirent.name) === '.ts') {
                const blocks = processFile(fullPath, threshold);
                const relativePath = path.relative(repoPath, fullPath);
                jsonlFile.write(JSON.stringify({ filename: relativePath, ...blocks }) + '\n');
            }
        });
    }

    processDirectory(repoPath);
    jsonlFile.end();
}

// 处理一个大文件夹中的所有 TypeScript 仓库，并将结果保存到指定文件夹
function processRepositories(rootPath: string, outputDir: string, threshold: number): void {
    fs.readdirSync(rootPath, { withFileTypes: true }).forEach(dirent => {
        if (dirent.isDirectory()) {
            const repoPath = path.join(rootPath, dirent.name);
            processRepository(repoPath, outputDir, threshold);
        }
    });
}

// 示例使用
const rootPath = 'data';
const outputDir = 'save';
const threshold = 5;
processRepositories(rootPath, outputDir, threshold);

"""

# 设置窗口大小和滑动大小
window_size = 10
slide_size = 8

# 获取分块结果
chunks = chunk_code(source_code, window_size, slide_size)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
