"""inspect_simple_clean_predictions.py

在 simple_clean 数据集上查看当前 best_simple_clean_model.npz 的具体预测情况。
只使用神经网络模型本身（不经过混合规则），方便观察模型学到了什么。

用法：
    python scripts/inspect_simple_clean_predictions.py [config_path] [num_samples]

默认：
    config_path = configs/simple_clean_config.json
    num_samples = 10
"""

import json
import sys
from pathlib import Path


# 设置项目根目录，并加入 src 到 Python 路径
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel


def load_config(config_path: str):
    cfg_file = project_root / config_path
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✅ 配置加载成功: {cfg_file}")
        return config
    except Exception as e:
        print(f"❌ 加载配置失败 ({cfg_file}): {e}")
        return None


def inspect_predictions(config_path: str = "configs/simple_clean_config.json", num_samples: int = 10):
    config = load_config(config_path)
    if config is None:
        return

    # 初始化 tokenizer 与数据
    tokenizer = Tokenizer()
    data_cfg = config["data"]
    val_path = project_root / data_cfg["val_path"]
    max_val = data_cfg.get("max_val_samples", num_samples)

    val_data = load_dataset(str(val_path), tokenizer, max_val)
    if not val_data:
        print(f"❌ 验证集数据为空或加载失败: {val_path}")
        return

    # 初始化并加载模型
    model_cfg = config["model"]
    paths_cfg = config["paths"]
    model = ImprovedSimpleModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_cfg["hidden_size"],
        max_length=model_cfg["max_length"],
        learning_rate=model_cfg["learning_rate"],
    )

    model_path = project_root / paths_cfg["model_save_path"]
    if not model.load_model(str(model_path)):
        print(f"⚠️ 无法从 {model_path} 加载模型，使用随机权重进行预测。")

    print("\n=== simple_clean 验证集预测样例（纯神经网络）===")
    show_n = min(num_samples, len(val_data))

    exact_matches = 0
    for idx, sample in enumerate(val_data[:show_n], start=1):
        input_text = sample["input_text"]
        target_text = sample["target_text"].strip()
        original_prop = sample.get("original_prop", "")

        predicted_tokens = model.predict(sample["input"], tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens).strip()

        is_exact = predicted_text == target_text
        if is_exact:
            exact_matches += 1

        print("\n------------------------------")
        print(f"样本 {idx}:")
        if original_prop:
            print(f"  original_prop        : {original_prop}")
        print(f"  noisy_prop (input)   : {input_text}")
        print(f"  target_contrapositive: {target_text}")
        print(f"  predicted            : {predicted_text}")
        print(f"  exact_match          : {is_exact}")

    print("\n=== 小结 ===")
    print(f"展示样本数: {show_n}")
    if show_n:
        acc = exact_matches / show_n
    else:
        acc = 0.0
    print(f"其中 exact match 数: {exact_matches} ({acc:.2%})")


def main():
    # 命令行参数：config_path, num_samples
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/simple_clean_config.json"

    if len(sys.argv) > 2:
        try:
            num_samples = int(sys.argv[2])
        except ValueError:
            num_samples = 10
    else:
        num_samples = 10

    inspect_predictions(config_path, num_samples)


if __name__ == "__main__":
    main()

