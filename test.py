from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.test_utils.testing import get_backend
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM
from pypinyin import lazy_pinyin, Style


yunjiao2id = { 
    'a': 0, 'ia': 0, 'ua': 0, 'va': 0, 'üa': 0, 
    'e': 1, 'o': 1, 'uo': 1, 'ie': 1, 'ue': 1, 'üe': 1, 've': 1, 
    'u': 2, 
    'i': 3, 'ü': 3, 'v': 3, 
    'ai': 4, 'uai': 4, 
    'ao': 5, 'iao': 5, 
    'ou': 6, 'iu': 6, 'iou': 6, 
    'an': 7, 'ian': 7, 'uan': 7, 'üan': 7, 'van': 7, 
    'en': 8, 'in': 8, 'un': 8, 'ün': 8, 'vn': 8, 
    'ang': 9, 'iang': 9, 'uang': 9, 
    'eng': 10, 'ing': 10, 'ueng': 10, 'ong': 10, 'iong': 10, 
    'er': 11, 
    'ei': 12, 'ui': 12, 'uei': 12, 'vei': 12
}

def get_yunjiao(token):
    # 获取token的韵母，如果不是中文则返回[UNK]
    if '\u4e00' <= token <= '\u9fa5':
        pinyin = lazy_pinyin(token, style=Style.FINALS)
        final = pinyin[-1] if pinyin else '[UNK]'
        return final
    else:
        return '[UNK]'

def get_yunid(token):
    final = get_yunjiao(token)
    return yunjiao2id.get(final, -1)  # 当找不到韵脚时，返回 -1 作为特殊标记
    

class RhymeLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, rhyme_alpha: float, n_gram_size: int=1):
        self.tokenizer = tokenizer
        self.rhyme_alpha = rhyme_alpha
        self.n_gram_size = n_gram_size

        if not isinstance(rhyme_alpha, (int, float)) or not (rhyme_alpha > 0):
            raise ValueError(f"`rhyme_alpha` has to be a strictly positive integer or float, but is {rhyme_alpha}") 
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenid2yunid = torch.tensor(
            [get_yunid(self.tokenizer.decode(token_id)) for token_id in range(self.tokenizer.vocab_size)],
            device=self.device
        )

    
    def _find_last_two_positions(self,input_ids, token_id=198):
        """
        input_ids: (batch_size, seq_len)
        返回每个样本最近两次token_id出现的位置（没有则为-1）
        """
        batch_size, _ = input_ids.shape
        mask = (input_ids == token_id)  # (batch_size, seq_len)，True表示token_id出现
        positions = [torch.where(mask[i])[0] for i in range(batch_size)]
        # 取最后两个出现的位置
        last_pos = torch.tensor([p[-1].item() if len(p) > 0 else -1 for p in positions], device=input_ids.device)
        last2_pos = torch.tensor([p[-2].item() if len(p) > 1 else -1 for p in positions], device=input_ids.device)
        return last_pos, last2_pos
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        在每个生成步骤被调用
        input_ids: (batch_size, sequence_length)，已经生成的token
        scores: (batch_size, vocab_size)，当前步的logits
        """
        batch_size, seq_len = input_ids.shape
        pad_token_id = self.tokenizer.pad_token_id
        last_pos, last2_pos = self._find_last_two_positions(input_ids, token_id=198)
        print(last_pos,last2_pos)
        # 如果 last_pos 和 last2_pos 存在无效，直接返回原始 scores
        is_valid_pos = (last_pos != -1) & (last2_pos != -1)
        if not is_valid_pos.all(): 
            return scores
        
        # 计算参考词的位置
        is_pad = (input_ids == pad_token_id)
        pad_count = is_pad.sum(dim=1) 
        actual_len = seq_len - pad_count  # (batch_size,)
        ref_position = last2_pos + actual_len - last_pos  # (batch_size,)

        # 获取参考词的韵母id
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        ref_token_id = input_ids[batch_indices, ref_position]  # (batch_size,)
        vocab_size_pad_len = scores.shape[-1] - tokenizer.vocab_size  # 由于模型参数一般会设置为128的倍数，这里pad到和scores一样的长度
        all_yunids = F.pad(self.tokenid2yunid, (0, vocab_size_pad_len), value=-1)  # (vocab_size,)
        ref_yunids = all_yunids[ref_token_id]   # (batch_size,)
        print(self.tokenizer.decode(ref_token_id))

       # 判断韵母是否匹配
        ref_yunids_expanded = ref_yunids.unsqueeze(1).expand_as(scores)  # (batch_size, vocab_size)
        rhyme_match_matrix = (all_yunids.unsqueeze(0) == ref_yunids_expanded)  # (batch_size, vocab_size)
    
        # 判断是否在n_gram内
        distance = actual_len - last_pos  # (batch_size,)
        within_n_gram = distance <= self.n_gram_size  # (batch_size,)
        within_n_gram_matrix = within_n_gram.unsqueeze(1).expand_as(scores)  # (batch_size, vocab_size)

        # 调整scores
        adjusted_scores = torch.where(
            rhyme_match_matrix & within_n_gram_matrix,
            torch.where(scores >= 0, scores * self.rhyme_alpha, scores / self.rhyme_alpha),
            scores
        )  # (batch_size, vocab_size)

        max_per_batch, max_indices = adjusted_scores.max(dim=1)
        print(' '*20,self.tokenizer.decode(max_indices[0]))
        return adjusted_scores


device, _, _ = get_backend() 
model_id = "/data/project/model_weights/lyrics_gen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

rhyme_alpha = 100
rhyme_processor = RhymeLogitsProcessor(
    tokenizer=tokenizer,
    rhyme_alpha=rhyme_alpha
)

# 2. 准备输入
prompt_text = ["请你不要思考，直接以下面句子为开头写一首歌词，不同句子之间用换行来分割，不要加标点符号，句子开始：\n深夜里键盘声还在不停回响\n勇敢的探险家在最深处寻访\n"]
               
# "请你不要思考,直接以下面句子为开头写一首歌词：小时候世界像童话,渴望着自由的长大,"
encodings = tokenizer(prompt_text, padding=True, return_tensors="pt")
input_ids = encodings.input_ids.to(device)
attention_mask = encodings.attention_mask.to(device)
# output = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=50)
output = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=200, logits_processor=[rhyme_processor],repetition_penalty=1.2)
generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in output]

for text in generated_texts:
    print(text)

# 勇敢 的 探险 家   在   最   深处  寻 访,
# 山   河 之间 留下 白   玫瑰 路    痕 旁

# 目前的缺点：
# 1）歌词句子之间最好有显式分隔符，否则 find_last_two_positions 很容易出错（当然格式遵循问题可以通过微调缓解）
# 2）词表问题：依旧会在全token范围内去预测，而不是在单个汉字的词表内预测，导致无法控制长度
# 3）应该还是要 right-to-left 进行建模
# 4）效果不稳定：（1）长文本效果不行，因为主要来源于问题1，（2）生成结果容易重复，需要找 rhyme 和 repetition 的超参数 
