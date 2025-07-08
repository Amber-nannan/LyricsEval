import os
import numpy as np
from pypinyin import Style, lazy_pinyin
import jieba
from collections import Counter
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


PUNCS = set([",", ".", "?", "!", ":", "，", "。", "？", "！", "："])

yunjiaos = {
            "0":["a", "ia", "ua", "va", "üa"],
            "1":["e", "o", "uo", "ie", "ue", "üe", "ve"],
            "2":["u"],
            "3":["i", "ü", "v"],
            "4":["ai", "uai"],
            "5":["ao", "iao"],
            "6":["ou", "iu", "iou"],
            "7":["an", "ian", "uan", "üan", "van"],
            "8":["en", "in", "un", "ün", "vn"],
            "9":["ang", "iang", "uang"],
            "10":["eng", "ing", "ueng", "ong", "iong"],
            "11":["er"],
            "12":["ei", "ui", "uei", "vei"],
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
    
    def calculate_ppl(self, text: str, stride: int = 512) -> float:
        """计算困惑度 """
        if not self.model or not self.tokenizer:
            print("警告: 模型未加载，无法计算PPL")
            return -1.0
        
        try:
            encodings = self.tokenizer(text, return_tensors="pt")
            
            if hasattr(self.model.config, 'n_positions'):
                max_length = self.model.config.n_positions
            elif hasattr(self.model.config, 'max_position_embeddings'):
                max_length = self.model.config.max_position_embeddings
            else:
                max_length = 1024  # 默认值
            
            seq_len = encodings.input_ids.size(1)
            
            # 如果序列长度小于等于最大长度，直接计算
            if seq_len <= max_length:
                input_ids = encodings.input_ids.to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    ppl = torch.exp(loss).item()
                return ppl
            
            # 对于长序列，使用滑动窗口方法（歌词基本不会超过最大长度）
            nll_sum = 0.0
            n_tokens = 0
            prev_end_loc = 0
            
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss
                
                # 累积总的负对数似然和总的token数
                num_valid_tokens = (target_ids != -100).sum().item()  # target_ids中有效token的数量
                batch_size = target_ids.size(0)
                num_loss_tokens = num_valid_tokens - batch_size  # 由于内部标签移位，减去batch_size
                
                if num_loss_tokens > 0:
                    nll_sum += neg_log_likelihood * num_loss_tokens
                    n_tokens += num_loss_tokens
                
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
            
            if n_tokens == 0:
                return -1.0
            
            avg_nll = nll_sum / n_tokens
            ppl = torch.exp(avg_nll).item()
            
            return ppl
            
        except Exception as e:
            print(f"PPL计算错误: {e}")
            return -1.0
    
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
        """计算完整性，用每句最后一个token的PPL来计算"""
        
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
    
    def evaluate_lyrics(self, text: str, target_length: Optional[List[int]] = None, 
                        trigger_words: Optional[List[str]] = None,
                       original_text: Optional[str] = None) -> Dict[str, float]:
        """评估歌词"""
        results = {}
        results['ppl'] = self.calculate_ppl(text)
        results['distinct_1'] = self.calculate_distinct_n(text, 1)
        results['distinct_2'] = self.calculate_distinct_n(text, 2)
        results['completeness'] = self.calculate_completeness(text)
        results['rhyme_accuracy'] = self.calculate_rhyme_accuracy(text)
        if target_length:
            results['sentence_length_accuracy'] = self.calculate_sentence_length_accuracy(text, target_length)
        
        if trigger_words or original_text:
            results['trigger_word_effect'] = self.calculate_trigger_word_effect(text, original_text, trigger_words)

        return results
    
    def print_evaluation_report(self, results: Dict[str, float]):
        """打印评估报告"""
        print("=" * 50)
        print("歌词评估报告")
        print("=" * 50)
        for metric, score in results.items():
            print(f"{metric}: {score:.3f}")
        
        print("=" * 50)


# 使用示例
if __name__ == "__main__":
    evaluator = LyricsEvaluator(model_path="/data/project/model_weights/lyrics_gen/Qwen3-0.6B")

    sample_lyrics = """
    电视机响床头灯会亮
    手提包装心里还在想
    打开车窗看天又快亮
    房间空荡一人躺椅上
    没有光线很晃
    只是灯光太像
    关门后的小伤
    等待下班铃响
    """

    sample_lyrics = sample_lyrics.strip().replace("    ", "")

    results = evaluator.evaluate_lyrics(
        text=sample_lyrics,
        target_length=[9,9,9,9,6,6,6,6],
        trigger_words=["夜晚", "等待", "下班"]
    )

    evaluator.print_evaluation_report(results)



