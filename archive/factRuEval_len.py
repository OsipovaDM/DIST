import os
from transformers import AutoTokenizer
from config import DATA_DIR, BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
fact_dir = os.path.join(DATA_DIR, "factRuEval-2016-master")

results = []
for split in ("devset", "testset"):
    split_dir = os.path.join(fact_dir, split)
    if not os.path.isdir(split_dir):
        continue
    for fn in sorted(os.listdir(split_dir)):
        if fn.endswith(".txt"):
            path = os.path.join(split_dir, fn)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            chars = len(text)
            tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            results.append((split, fn, chars, tokens))

results.sort(key=lambda x: x[3], reverse=True)

print(f"{'split':<10}{'документ':<25}{'символы':>10}{'токены':>10}")
print("-" * 55)
for split, fn, chars, tokens in results[:15]:
    print(f"{split:<10}{fn:<25}{chars:>10}{tokens:>10}")

print(f"\nВсего документов: {len(results)}")
tokens_list = [r[3] for r in results]
tokens_list.sort()
print(f"MIN токенов:  {tokens_list[0]}")
print(f"MEDIAN:       {tokens_list[len(tokens_list)//2]}")
print(f"P95:          {tokens_list[int(0.95*len(tokens_list))]}")
print(f"MAX токенов:  {tokens_list[-1]}")
print(f"Документов > 256 токенов: {sum(1 for t in tokens_list if t > 256)} из {len(tokens_list)}")
print(f"Документов > 384 токенов: {sum(1 for t in tokens_list if t > 384)} из {len(tokens_list)}")
print(f"Документов > 512 токенов: {sum(1 for t in tokens_list if t > 512)} из {len(tokens_list)}")