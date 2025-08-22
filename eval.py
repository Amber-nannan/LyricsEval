import os
import numpy as np
from tqdm import tqdm
from pypinyin import Style, lazy_pinyin
import jieba
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataloader import LyricsDataset
from torch.utils.data import Dataset, DataLoader


PUNCS = set(["《", "》", ",", ".", "?", "!", ":", "，", "。", "？", "！", "："])

yunjiaos = {
    "0": ["e"],
    "1": ["ie", "ue", "ve"],
    "2": ["a", "ia", "ua"],
    "3": ["o", "uo"],
    "4": ["i"],
    "5": ["v"],
    "6": ["ei", "uei", "ui"],
    "7": ["ai", "uai"],
    "8": ["u"],
    "9": ["ou", "iu", "iou"],
    "10": ["ao", "iao"],
    "11": ["an", "uan"],
    "12": ["ian", "van"],
    "13": ["ang", "iang", "uang"],
    "14": ["in", "ing"],
    "15": ["en", "un", "uen"],
    "16": ["eng"],
    "17": ["iong", "ong"]
}


class LyricsEvaluator:
    """歌词评估类"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型用于PPL计算
        if model_path and os.path.exists(model_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()
                print(f"模型加载成功: {model_path}")
            except Exception as e:
                print(f"模型加载失败: {e}")
        
        # 韵脚映射表
        self.yunjiao2id = self._build_yunjiao2id()
    
    def _build_yunjiao2id(self) -> Dict[str, str]:
        """构建韵脚映射表"""
        yunjiao2id = {}
        for group_id, finals in yunjiaos.items():
            for final in finals:
                yunjiao2id[final] = group_id
        return yunjiao2id
    
    def calculate_ppl(self, texts: List[str], stride: int = 512) -> List[float]:
        """批量计算困惑度 """
        if not self.model or not self.tokenizer:
            print("警告: 模型未加载，无法计算PPL")
            return [-1.0] * len(texts)
        
        try:
            if hasattr(self.model.config, 'n_positions'):
                max_length = self.model.config.n_positions
            elif hasattr(self.model.config, 'max_position_embeddings'):
                max_length = self.model.config.max_position_embeddings   # qwen3 max_length= 40960
            else:
                max_length = 1024  # 默认值
            
            encodings = self.tokenizer(texts, padding=True, truncation=True, max_length = max_length, return_tensors="pt")
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                criterion = torch.nn.CrossEntropyLoss(reduction='none')  # 不进行平均，保留每个token的损失
                ppls = []
                # 第i个样本的困惑度 ppl_i = exp(mean(cross_entropy_loss_i))
                for i in range(input_ids.size(0)):
                    input_id = input_ids[i]
                    target = input_id 
                    logits_i = logits[i, :-1]  # 去掉最后一个token的预测，因为没有标签
                    target_i = target[1:]  # 去掉第一个token的标签，因为没有预测

                    # 计算当前样本的交叉熵损失、ppl
                    loss_i = criterion(logits_i, target_i)
                    loss_i = loss_i * attention_mask[i, 1:].float()  # 过滤掉填充部分
                    ppl = loss_i.sum().item() / attention_mask[i, 1:].sum().item()
                    ppls.append(ppl)
            return ppls

        except Exception as e:
            print(f"PPL计算错误: {e}")
            return [-1.0] * len(texts)
    
    def calculate_distinct_n(self, text: str, n: int) -> float:
        """计算distinct-n分数"""
        tokens = list(jieba.cut(text))
        # tokens = list(text)
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        return unique_ngrams / total_ngrams
    
    def calculate_completeness(self, text: str) -> float:
        """计算完整性，用每句最后一个token为结束符号的概率来计算"""
        
        if not self.model or not self.tokenizer:
            print("警告: 模型未加载，无法计算完整性")
            return -1.0
        
        try:
            lyrics = text.split('\n')
            total_log_likelihood = 0.0
            num_sentences = len(lyrics)

            if num_sentences == 0:
                print("错误: 文本中没有句子")
                return -1.0
            
            for sentence in lyrics:
                encodings = self.tokenizer(sentence, return_tensors="pt")
                input_ids = encodings.input_ids.to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits

                    # 获取最后一个token的预测概率
                    last_token_logits = logits[0, -1, :]  # logits的形状是 [batch_size, seq_len, vocab_size]，这里batch_size为1
                    last_token_probs = F.softmax(last_token_logits, dim=-1)

                    # 获取最后一个token为结束符号的概率
                    eos_token_id = self.tokenizer.eos_token_id
                    eos_token_prob = last_token_probs[eos_token_id]

                    # 负对数似然
                    neg_log_likelihood = -torch.log(eos_token_prob)
                    total_log_likelihood += neg_log_likelihood.item()

            # 负对数似然的平均的指数
            avg_neg_log_likelihood = total_log_likelihood / num_sentences
            completeness = torch.exp(torch.tensor(avg_neg_log_likelihood)).item()

            return completeness

        except Exception as e:
            print(f"完整性计算错误: {e}")
            return -1.0

    def get_rhyme_group(self, lyrics):
        """获取歌词的韵脚组"""
        try:
            lyrics_yun = []
            for sentence in lyrics:
                word = sentence[-1]
                pinyin = lazy_pinyin(word, style=Style.FINALS)
                if pinyin:
                    final = pinyin[0]
                    yunid = self.yunjiao2id.get(final, "unknown")
                    lyrics_yun.append(yunid)
            return lyrics_yun
        except Exception as e:
            print(f"歌词韵脚错误: {e}")
            return []

    def calculate_rhyme_accuracy(self, text: str, pattern: str = "AAAA") -> float:
        """ 计算歌词的押韵准确度, 押韵模式默认为AAAA """
        try:
            lyrics = text.split("\n")
            rhyme_group = self.get_rhyme_group(lyrics)
            
            if len(rhyme_group) < 2:
                return 0.0

            rhyme_count = 0
            total_count = 0

            if pattern == "AAAA":
                for i in range(0, len(rhyme_group) - 3):
                    if rhyme_group[i] == rhyme_group[i + 1] == rhyme_group[i + 2] == rhyme_group[i + 3]:
                        rhyme_count += 1
                    total_count += 1

            if pattern == "AABB":
                for i in range(0, len(rhyme_group) - 1, 2):
                    if i + 1 < len(rhyme_group) and rhyme_group[i] == rhyme_group[i + 1]:
                        rhyme_count += 1
                    total_count += 1

            if pattern == "ABAB":
                for i in range(0, len(rhyme_group) - 3, 2):
                    if rhyme_group[i] == rhyme_group[i + 2] and rhyme_group[i + 1] == rhyme_group[i + 3]:
                        rhyme_count += 1
                    total_count += 1

            return rhyme_count / total_count if total_count else 0

        except Exception as e:
            print(f"计算押韵准确度时出错: {e}")
            return 0.0

    def calculate_rhyme_count(self, text: str) -> float:
        """计算歌词的韵脚数量"""
        try:
            lyrics = text.split("\n")
            rhyme_group = self.get_rhyme_group(lyrics)
            return len(set(rhyme_group)-{"unknown"})

        except Exception as e:
            print(f"计算押韵数量时出错: {e}")
            return 0.0
        
    def calculate_sentence_length_accuracy(self, text: str, target_length: List[int], tolerance: int = 0) -> float:
        """计算句子长度准确率"""
        lyrics = text.split('\n')
        
        if not lyrics:
            return 0.0
        
        accurate_sentences = 0
        for i,sentence in enumerate(lyrics):
            s_length = len(sentence)
            t_length = target_length[i]
            if abs(s_length - t_length) <= tolerance:
                accurate_sentences += 1
        
        return accurate_sentences / len(lyrics)
        
    def calculate_trigger_word_effect(self, generated_text: str, original_text: Optional[str] = None,
                                    trigger_words: Optional[List[str]] = None) -> float:
        """计算触发词效果 (使用BLEU分数)"""
        try:
            if original_text:
                original_tokens = list(jieba.cut(original_text))
            elif trigger_words:
                original_tokens = trigger_words

            generated_tokens = list(jieba.cut(generated_text))
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([original_tokens], generated_tokens, 
                                     smoothing_function=smoothing)
            
            return bleu_score
        except Exception as e:
            print(f"BLEU计算错误: {e}")
            return 0.0
    
    def evaluate_lyrics(self, texts: List[str],
                        target_lengths: Optional[List[List[int]]] = None, 
                        trigger_wordss: Optional[List[List[str]]] = None,
                       original_texts: Optional[List[str]] = None
                       ) -> Dict[str, List[float]]:
        """评估歌词"""
        results = defaultdict(list)
        results['ppl'] = self.calculate_ppl(texts)

        for i, text in enumerate(texts):
            results['distinct_1'].append(self.calculate_distinct_n(text, 1))
            results['distinct_2'].append(self.calculate_distinct_n(text, 2))
            # results['completeness'].append(self.calculate_completeness(text))
            # results['rhyme_accuracy'].append(self.calculate_rhyme_accuracy(text))
            results['rhyme_count'].append(self.calculate_rhyme_count(text))

            if target_lengths and target_lengths[i]:
                results['sentence_length_accuracy'].append(self.calculate_sentence_length_accuracy(text, target_lengths[i]))
            
            if original_texts and original_texts[i]:
                results['trigger_word_effect'].append(self.calculate_trigger_word_effect(text, original_text = original_texts[i]))
            elif trigger_wordss and trigger_wordss[i]:
                results['trigger_word_effect'].append(self.calculate_trigger_word_effect(text, tringger_words = trigger_wordss[i]))
        return results


if __name__ == "__main__":
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate lyrics generation model.')
    parser.add_argument('--model_path', type=str, default='/data/project/model_weights/lyrics_gen/Qwen3-0.6B', help='Path to the model weights')
    parser.add_argument('--data_path', type=str, default='./data/AI/ai_lyrics.json', help='Path to data file')
    parser.add_argument('--output_path', type=str, default='./results/results.json', help='Path to output file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    args = parser.parse_args()

    # 读取数据
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # 简单的预处理
    import re
    lyrics = []
    for item in data:
        item['lyrics'] = [line.strip() for line in item['lyrics'] if len(line) > 1]
        item['lyrics'] = [line for line in item['lyrics'] if not re.findall('(《|》|主歌|副歌|间奏|尾奏|桥段|独奏)',line)]
        lyrics.append('\n'.join(item['lyrics']))
    
    # 创建数据集和数据加载器、评估器
    dataset = LyricsDataset(lyrics)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    evaluator = LyricsEvaluator(args.model_path)

    # 评估
    results = defaultdict(list)
    for batch in tqdm(dataloader, desc="Evaluating Lyrics", unit="batch"):
        lyrics = batch['lyrics']
        target_length = batch['target_length'] if 'target_length' in batch else None
        trigger_words = batch['trigger_words'] if 'trigger_words' in batch else None
        
        batch_res = evaluator.evaluate_lyrics(
            texts=lyrics,
            target_lengths=target_length,
            trigger_wordss=trigger_words
        )

        for key, values in batch_res.items():
            results[key].extend(values)
    
    # 保存结果
    with open(args.output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Eval results saved to", args.output_path)



