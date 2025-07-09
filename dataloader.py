import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

class LyricsDataset(Dataset):
    def __init__(self, lyrics: List[str], target_lengths: Optional[List[List[int]]]=None, 
                trigger_words: Optional[List[List[str]]]=None):
        self.lyrics = lyrics
        self.target_lengths = target_lengths
        self.trigger_words = trigger_words

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        data = {'lyrics': self.lyrics[idx]}
        
        if self.target_lengths is not None and self.target_lengtShs[idx]:
            data['target_length'] = self.target_lengths[idx]
        
        if self.trigger_words is not None and self.trigger_words[idx]:
            data['trigger_words'] = self.trigger_words[idx]
        
        return data



if __name__ == "__main__":
    import json

    with open('/data/project/project_siting/LyricsEval/data/not_AI/cleaded_lhz_wyy_song_meta_0320.json', 'r') as f:
        data = json.load(f)
    
    lyrics = []
    for item in data:
        lyrics.append('\n'.join(item['lyrics']))
    
    lyrics_data = LyricsDataset(lyrics)