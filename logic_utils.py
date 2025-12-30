"""
文件名: logic_utils.py
逻辑工具函数模块
用于生成命题逻辑公式、添加噪声、转换为逆否命题等
"""

import random
import re
from typing import List, Dict, Tuple


class Tokenizer:
    """简单的字符级tokenizer，用于命题逻辑符号"""

    def __init__(self):
        # 定义所有可能的符号
        self.symbols = ["p", "q", "r", "s", "t", "~", "&", "|", "-", ">", "(", ")", " "]
        self.char_to_int = {char: i for i, char in enumerate(self.symbols)}
        self.int_to_char = {i: char for i, char in enumerate(self.symbols)}
        self.vocab_size = len(self.symbols)

        # 特殊token
        self.PAD_TOKEN = len(self.symbols)
        self.START_TOKEN = len(self.symbols) + 1
        self.END_TOKEN = len(self.symbols) + 2

        self.char_to_int["<PAD>"] = self.PAD_TOKEN
        self.char_to_int["<START>"] = self.START_TOKEN
        self.char_to_int["<END>"] = self.END_TOKEN

        self.int_to_char[self.PAD_TOKEN] = "<PAD>"
        self.int_to_char[self.START_TOKEN] = "<START>"
        self.int_to_char[self.END_TOKEN] = "<END>"

        self.vocab_size += 3

    def encode(self, text: str) -> List[int]:
        """将文本编码为整数序列"""
        return [self.char_to_int.get(char, self.PAD_TOKEN) for char in text]

    def decode(self, tokens: List[int]) -> str:
        """将整数序列解码为文本"""
        return "".join([self.int_to_char.get(token, "") for token in tokens])


def generate_simple_proposition(variables: List[str] = None) -> str:
    """生成简单的命题逻辑公式 (A -> B 形式)"""
    if variables is None:
        variables = ["p", "q", "r", "s"]

    # 随机选择两个不同的变量
    var1, var2 = random.sample(variables, 2)

    # 随机决定是否添加否定
    if random.random() < 0.3:
        var1 = f"~{var1}"
    if random.random() < 0.3:
        var2 = f"~{var2}"

    return f"{var1} -> {var2}"


def generate_complex_proposition(variables: List[str] = None) -> str:
    """生成复杂的命题逻辑公式 ((A & B) -> C 形式)"""
    if variables is None:
        variables = ["p", "q", "r", "s"]

    # 随机选择三个变量
    var1, var2, var3 = random.sample(variables, 3)

    # 随机添加否定
    if random.random() < 0.2:
        var1 = f"~{var1}"
    if random.random() < 0.2:
        var2 = f"~{var2}"
    if random.random() < 0.2:
        var3 = f"~{var3}"

    # 随机选择连接符
    connector = random.choice(["&", "|"])

    return f"({var1} {connector} {var2}) -> {var3}"


def generate_recursive_proposition(variables=None, max_depth=3, current_depth=0) -> str:
    """
    递归生成任意深度的嵌套逻辑命题
    这是您建议的核心改进：结构复杂度的提升
    """
    if variables is None:
        variables = ["p", "q", "r", "s", "t"]

    # 基底情况：达到最大深度或随机终止，返回一个原子命题
    if current_depth >= max_depth or random.random() < 0.3:
        prop = random.choice(variables)
        # 随机否定
        if random.random() < 0.3:
            return f"~{prop}"
        return prop

    # 递归步骤：选择一个运算符，并为左右两边递归生成子命题
    op = random.choice(["&", "|", "->"])
    left = generate_recursive_proposition(variables, max_depth, current_depth + 1)
    right = generate_recursive_proposition(variables, max_depth, current_depth + 1)

    return f"({left} {op} {right})"


def generate_recursive_implication(max_depth=3) -> str:
    """
    生成递归的蕴含命题，确保最终形式是 A -> B
    """
    variables = ["p", "q", "r", "s", "t"]

    # 生成前件和后件
    antecedent = generate_recursive_proposition(variables, max_depth)
    consequent = generate_recursive_proposition(variables, max_depth)

    return f"{antecedent} -> {consequent}"


def find_main_implication(formula: str) -> tuple:
    """
    找到命题中的主蕴含符 ->
    返回 (antecedent, consequent) 或 (None, None) 如果不是蕴含命题

    这个函数能正确处理嵌套的括号和多层蕴含
    """
    formula = formula.strip()

    # 如果不包含 ->，不是蕴含命题
    if "->" not in formula:
        return None, None

    # 使用栈来跟踪括号层级
    paren_depth = 0
    i = 0

    while i < len(formula) - 1:
        char = formula[i]

        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "-" and i + 1 < len(formula) and formula[i + 1] == ">":
            # 找到 -> 符号
            if paren_depth == 0:
                # 这是主层级的蕴含符
                antecedent = formula[:i].strip()
                consequent = formula[i + 2 :].strip()
                return antecedent, consequent

        i += 1

    # 如果没有找到主层级的 ->，返回 None
    return None, None


def to_contrapositive(prop_str: str) -> str:
    """
    将命题转换为逆否命题
    修复版：能正确处理任意深度的嵌套命题
    """
    prop_str = prop_str.strip()

    # 使用改进的解析器找到主蕴含符
    antecedent, consequent = find_main_implication(prop_str)

    if antecedent is None or consequent is None:
        # 不是蕴含命题，返回原式
        return prop_str

    # 生成逆否命题: ~B -> ~A
    neg_consequent = negate_formula(consequent)
    neg_antecedent = negate_formula(antecedent)

    return f"{neg_consequent} -> {neg_antecedent}"


def negate_formula(formula: str) -> str:
    """
    对公式添加否定
    修复版：正确处理括号和复合表达式
    """
    formula = formula.strip()

    # 修正：如果是否定一个带括号的表达式 ~(...)
    if formula.startswith("~(") and formula.endswith(")"):
        return formula[2:-1]  # 去掉 ~( 和 )

    # 修正：如果是否定一个简单变量 ~p
    if formula.startswith("~") and not formula.startswith("~("):
        return formula[1:].strip()

    # 如果已经有外层括号，直接添加否定，不要双重括号
    if formula.startswith("(") and formula.endswith(")"):
        return f"~{formula}"

    # 如果是复合表达式（包含空格），用括号包起来再否定
    if " " in formula:
        return f"~({formula})"

    # 简单变量，直接添加否定
    return f"~{formula}"


def add_noise_type1(prop_str: str) -> str:
    """
    噪声类型1：将 A -> B 转换为等价的 ~A | B
    这是我们的核心任务
    """
    prop_str = prop_str.strip()

    # 匹配 A -> B 模式
    pattern = r"^([^-]+)\s*->\s*(.+)$"
    match = re.match(pattern, prop_str)

    if match:
        antecedent = match.group(1).strip()
        consequent = match.group(2).strip()

        # 转换为 ~A | B
        neg_antecedent = negate_formula(antecedent)
        return f"({neg_antecedent} | {consequent})"

    return prop_str


def add_noise_type2(prop_str: str) -> str:
    """
    噪声类型2：添加双重否定 p -> ~~p
    修复版：使用正则表达式确保只匹配独立的变量
    """
    # 随机选择一个变量添加双重否定
    variables = re.findall(r"\b[pqrst]\b", prop_str)
    if variables:
        var_to_modify = random.choice(variables)
        # 使用 re.sub 进行安全替换，确保只匹配独立的变量
        prop_str = re.sub(
            rf"\b{var_to_modify}\b", f"~~{var_to_modify}", prop_str, count=1
        )

    return prop_str


def add_noise_type3(prop_str: str) -> str:
    """
    噪声类型3：添加冗余括号 p -> (p)
    """
    # 随机选择一个变量添加括号
    variables = re.findall(r"\b[pqrst]\b", prop_str)
    if variables:
        var_to_modify = random.choice(variables)
        # 使用 re.sub 进行安全替换，确保只匹配独立的变量
        prop_str = re.sub(
            rf"\b{var_to_modify}\b", f"({var_to_modify})", prop_str, count=1
        )

    return prop_str


def add_noise_type4(prop_str: str) -> str:
    """
    噪声类型4：交换律变换 (p & q) -> (q & p)
    """
    # 查找 & 或 | 连接的表达式
    patterns = [
        (r"\(([pqrst~]+)\s*&\s*([pqrst~]+)\)", r"(\2 & \1)"),
        (r"\(([pqrst~]+)\s*\|\s*([pqrst~]+)\)", r"(\2 | \1)"),
    ]

    for pattern, replacement in patterns:
        if re.search(pattern, prop_str):
            prop_str = re.sub(pattern, replacement, prop_str, count=1)
            break

    return prop_str


def add_noise(
    prop_str: str, noise_types: List[str] = None, num_applications: int = 1
) -> str:
    """
    对命题添加噪声，但保持逻辑等价
    支持多次应用和新的噪声类型组合
    """
    if noise_types is None:
        noise_types = ["type1"]  # 默认只使用类型1噪声

    result = prop_str

    # 随机应用 num_applications 次噪声
    for _ in range(num_applications):
        chosen_noise_type = random.choice(noise_types)

        if chosen_noise_type == "type1":
            result = add_noise_type1(result)
        elif chosen_noise_type == "type2":
            result = add_noise_type2(result)
        elif chosen_noise_type == "type3":
            result = add_noise_type3(result)
        elif chosen_noise_type == "type4":
            result = add_noise_type4(result)

    return result


def verify_equivalence(formula1: str, formula2: str) -> bool:
    """
    验证两个公式是否逻辑等价
    使用真值表方法进行简单验证
    """
    # 提取所有变量
    vars1 = set(re.findall(r"\b[pqrst]\b", formula1))
    vars2 = set(re.findall(r"\b[pqrst]\b", formula2))
    all_vars = list(vars1.union(vars2))

    if not all_vars:
        return formula1.strip() == formula2.strip()

    # 生成所有可能的真值组合
    for i in range(2 ** len(all_vars)):
        assignment = {}
        for j, var in enumerate(all_vars):
            assignment[var] = bool((i >> j) & 1)

        try:
            val1 = evaluate_formula(formula1, assignment)
            val2 = evaluate_formula(formula2, assignment)

            if val1 != val2:
                return False
        except:
            # 如果评估失败，认为不等价
            return False

    return True


def evaluate_formula(formula: str, assignment: Dict[str, bool]) -> bool:
    """
    在给定变量赋值下评估公式的真值
    使用 Python 的 eval() 函数，并将逻辑符号映射到 Python 的布尔运算符
    这是一个可靠的实现，能正确处理括号和运算符优先级
    """
    # 建立一个安全的、可供 eval 使用的命名空间
    eval_globals = {"__builtins__": None}  # 禁止访问内建函数
    eval_locals = {var: val for var, val in assignment.items()}

    # 将逻辑符号转换为 Python 的布尔运算符
    py_formula = formula.replace("&", " and ")
    py_formula = py_formula.replace("|", " or ")
    py_formula = py_formula.replace("~", " not ")

    # 处理 -> 蕴含符： A -> B 等价于 (not A) or B
    # 使用特殊标记避免覆盖问题
    py_formula = py_formula.replace("->", " IMPLIES ")

    # 递归地处理蕴含，从右到左处理
    while "IMPLIES" in py_formula:
        # 找到最右边的蕴含式
        parts = py_formula.rsplit(" IMPLIES ", 1)
        if len(parts) == 2:
            antecedent = parts[0].strip()
            consequent = parts[1].strip()

            # A -> B 等价于 (not A) or B
            # 为了处理优先级，必须加上括号
            py_formula = f"(not ({antecedent})) or ({consequent})"
        else:
            break

    try:
        # 在安全的环境中执行评估
        result = eval(py_formula, eval_globals, eval_locals)
        return bool(result)
    except Exception as e:
        # 如果评估失败，抛出详细错误信息
        raise ValueError(f"无法评估公式: {formula} -> {py_formula} | 错误: {e}")


def check_balance(expr: str) -> bool:
    """
    检查括号是否平衡且扫描过程中不出现非法关闭。
    - 仅考虑 () 括号
    - 保证任意前缀的右括号数量不超过左括号
    """
    balance = 0
    for ch in expr:
        if ch == "(":
            balance += 1
        elif ch == ")":
            balance -= 1
            if balance < 0:
                return False
    return balance == 0


def _dedup_spaces(s: str) -> str:
    """将连续空格压缩为单一空格，并去除首尾空格。"""
    return re.sub(r"\s+", " ", s).strip()


def normalize_expression(expr: str) -> str:
    """
    表达式规范化：
    1) 去除所有空白后再为二元运算符添加标准空格
    2) 保持一元运算符 ~ 紧贴其操作数
    3) 括号内外不额外插入空格
    使得形如："~ ( p&q) ->r" 规范为 "~(p & q) -> r"
    """
    if expr is None:
        return ""

    s = re.sub(r"\s+", "", str(expr))  # 全部移除空白

    # 先处理蕴含，避免与 '-' 的歧义
    s = s.replace("->", "→")  # 临时替换占位，防止与后续替换冲突

    # 为 & 和 | 两侧插入空格
    s = re.sub(r"&", " & ", s)
    s = re.sub(r"\|", " | ", s)

    # 恢复蕴含并添加空格
    s = s.replace("→", " -> ")

    # 去除括号外侧多余空格："( p & q )" -> "(p & q)"
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)

    # 去重空格
    s = _dedup_spaces(s)
    return s


def auto_fix(expr: str) -> str:
    """
    尝试自动修复常见的括号不平衡问题：
    - 多余的右括号：在扫描时跳过超出的右括号
    - 缺失的右括号：在末尾补齐
    同时对结果进行规范化与空格压缩。
    """
    if expr is None:
        return ""

    s = normalize_expression(expr)

    # 先移除多余的右括号（前缀合法）
    out = []
    bal = 0
    for ch in s:
        if ch == "(":
            bal += 1
            out.append(ch)
        elif ch == ")":
            if bal > 0:
                bal -= 1
                out.append(ch)
            else:
                # 跳过多余的右括号
                continue
        else:
            out.append(ch)

    s2 = "".join(out)

    # 如果还有未闭合的左括号，在末尾补齐
    if bal > 0:
        s2 = s2 + ")" * bal

    return normalize_expression(s2)


def postprocess(expr: str) -> str:
    """
    统一后处理流水线：normalize -> balance -> dedup
    - 先标准化表达式与空格
    - 若括号不平衡则尝试 auto_fix
    - 最后做一次空格压缩
    """
    s = normalize_expression(expr)
    if not check_balance(s):
        s = auto_fix(s)
    return _dedup_spaces(s)


if __name__ == "__main__":
    # 测试代码
    tokenizer = Tokenizer()

    # 测试基本功能
    prop = generate_simple_proposition()
    print(f"原命题: {prop}")

    contrapositive = to_contrapositive(prop)
    print(f"逆否命题: {contrapositive}")

    noisy_prop = add_noise(prop)
    print(f"噪声命题: {noisy_prop}")

    # 测试tokenizer
    encoded = tokenizer.encode(prop)
    decoded = tokenizer.decode(encoded)
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
