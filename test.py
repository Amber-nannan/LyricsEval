from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.test_utils.testing import get_backend
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM
from pypinyin import lazy_pinyin, Style

# yunjiaos = {
#             "0":["a", "ia", "ua", "va", "üa"],
#             "1":["e", "o", "uo", "ie", "ue", "üe", "ve"],
#             "2":["u"],
#             "3":["i", "ü", "v"],
#             "4":["ai", "uai"],
#             "5":["ao", "iao"],
#             "6":["ou", "iu", "iou"],
#             "7":["an", "ian", "uan", "üan", "van"],
#             "8":["en", "in", "un", "ün", "vn"],
#             "9":["ang", "iang", "uang"],
#             "10":["eng", "ing", "ueng", "ong", "iong"],
#             "11":["er"],
#             "12":["ei", "ui", "uei", "vei"],
#            }



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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenid2yunid = torch.tensor(
            [get_yunid(self.tokenizer.decode(token_id)) for token_id in range(self.tokenizer.vocab_size)],
            device=self.device
        )
        print(self.tokenid2yunid[109949])
    
    def _find_last_two_positions(self,input_ids, token_id=11):
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
        print(input_ids)
        batch_size, seq_len = input_ids.shape
        pad_token_id = self.tokenizer.pad_token_id
        last_pos, last2_pos = self._find_last_two_positions(input_ids, token_id=11)
        print(last2_pos)
        print(last_pos)
        
        # 计算参考词的位置
        is_pad = (input_ids == pad_token_id)
        pad_pos = torch.where(is_pad, torch.arange(seq_len, device=input_ids.device), seq_len)
        first_pad_pos = pad_pos.min(dim=1).values  # (batch_size,)
        actual_lengths = torch.where(first_pad_pos == seq_len, seq_len, first_pad_pos)  # (batch_size,)
        ref_position = last2_pos + actual_lengths - last_pos  # (batch_size,)
        print(ref_position)

        # 获取参考词的韵母id
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        ref_token_id = input_ids[batch_indices, ref_position]  # (batch_size,)
        ref_yunids = self.tokenid2yunid[ref_token_id]   # (batch_size,)

       # 计算韵母匹配的bonus
        ref_yunids_expanded = ref_yunids.unsqueeze(1).expand_as(scores)  # (batch_size, vocab_size)
        # scores.shape[-1]=151936 不等于 tokenizer.vocab_size=151643
        # 因为计算效率原因，模型参数一般会设置为128的倍数，这里我们通过在右侧填充-1 统一vocab_size
        pad_len = scores.shape[-1] - tokenizer.vocab_size  
        all_yunids = F.pad(self.tokenid2yunid, (0, pad_len), value=-1)  # (vocab_size,)
        rhyme_match_matrix = (all_yunids.unsqueeze(0) == ref_yunids_expanded)  # (batch_size, vocab_size)
        rhyme_bonus = torch.where(rhyme_match_matrix, self.rhyme_alpha, 1.0)  # (batch_size, vocab_size)
        adjusted_scores = scores * rhyme_bonus
        max_per_batch, max_indices = scores.max(dim=1)
        print(max_per_batch)    # shape: (batch_size,)
        print(max_indices)      # shape: (batch_size,)
        # adjusted_scores也求最大值
        print('='*50)
        max_per_batch, max_indices = adjusted_scores.max(dim=1)
        print(max_per_batch)    # shape: (batch_size,)
        print(max_indices)      # shape: (batch_size,)
        print(self.tokenizer.decode(max_indices[0]))
        print(self.tokenizer.decode(max_indices[1]))
        return adjusted_scores


def test():
    device, _, _ = get_backend() 
    model_id = "/data/project/model_weights/lyrics_gen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rhyme_alpha = 1.1
    rhyme_processor = RhymeLogitsProcessor(
        tokenizer=tokenizer,
        rhyme_alpha=rhyme_alpha
    )


    input_ids = torch.tensor([[112720, 101041,  87752,  27091, 109949,  17714, 111749,  61443, 108462,
            114355,   5122, 109533,  69249, 112006,  70074, 104241, 101677,  18397,
            99365,     11, 114355,  99604, 109949, 101920,  85106,  71134,  22243,
                11],
            [112720, 101041,  87752,  27091, 109949,  17714, 111749,  61443, 108462,
            114355,   5122, 104284,  26232,  99787, 116176, 107693, 101281,     11,
            114355,  99604, 109949, 101920,  85106,  71134,  22243,     11, 151643,
            151643]], device='cuda:0')

    scores =  torch.ones(input_ids.shape[0], tokenizer.vocab_size, device=input_ids.device)
    res = rhyme_processor(input_ids,scores)
    print(res)


device, _, _ = get_backend() 
model_id = "/data/project/model_weights/lyrics_gen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

rhyme_alpha = 10000
rhyme_processor = RhymeLogitsProcessor(
    tokenizer=tokenizer,
    rhyme_alpha=rhyme_alpha
)

# 2. 准备输入
prompt_text = ["请你不要思考,直接以下面句子为开头写一首歌词：深夜里键盘声还在不停回响,勇敢的探险家在最深处寻访,",
               "请你不要思考,直接以下面句子为开头写一首歌词：偶尔会迷茫,偶尔会疲倦,"]
encodings = tokenizer(prompt_text, padding=True, return_tensors="pt")
input_ids = encodings.input_ids.to(device)
attention_mask = encodings.attention_mask.to(device)
# output = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=50)
output = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=50, logits_processor=[rhyme_processor])
generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in output]

for text in generated_texts:
    print(text)

# 勇敢 的 探险 家   在  最   深处  寻 访,
# 山   河 之间 留下 白  玫瑰 路    痕 旁


# 目前的缺点：
# 1）歌词句子之间最好有显式分隔符，否则prompt格式受限
# 2）词表问题：依旧会在全token范围内去预测，而不是在单个汉字的词表内预测
# 3）考虑改成 n-gram，不需要全句都押韵
# 4）应该还是要 right-to-left 进行建模


