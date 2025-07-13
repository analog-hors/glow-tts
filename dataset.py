import torch, numpy as np, torchaudio, librosa
from torch.utils.data import Dataset
from g2p_en import G2p

ARPABET_STRESS = ["0", "1", "2"]
ARPABET_VOWELS = [
    "AA", "AE", "AH", "AO", "AW",
    "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW",
]
ARPABET_CONSONANTS = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N",
    "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
]

TOKEN_PAD = "<pad>"
TOKEN_PUNCT = [" ", "!", "'", ",", "-", ".", "..", "?"]

TOKENS = [
    TOKEN_PAD,
    *TOKEN_PUNCT,
    *(v + s for v in ARPABET_VOWELS for s in ARPABET_STRESS),
    *ARPABET_CONSONANTS,
]

def tokenize(g2p: G2p, s: str) -> torch.Tensor:
    return torch.tensor([TOKENS.index(t) for t in g2p(s)])

def preprocess_ljspeech_dataset(path: str, out: str, mel_transform: torch.nn.Module):
    g2p = G2p()
    index: list[str] = []
    with open(f"{path}/metadata.csv") as file:
        for line in file:
            name, _text, norm_text = line.strip().split("|")
            index.append(name)

            encoded = tokenize(g2p, norm_text)
            encoded_np = encoded.numpy()

            waveform, _sample_rate = torchaudio.load(f"{path}/wavs/{name}.wav")
            waveform_np = waveform.squeeze(0).numpy()

            trimmed_np, _ = librosa.effects.trim(waveform_np, top_db=20)
            trimmed = torch.from_numpy(trimmed_np).unsqueeze(0)

            mels = mel_transform(trimmed).squeeze(0)
            mels_np = mels.numpy()

            np.save(f"{out}/{name}-text.npy", encoded_np, allow_pickle=False)
            np.save(f"{out}/{name}-mels.npy", mels_np, allow_pickle=False)
    
    with open(f"{out}/index.txt", "w+") as index_file:
        for line in index:
            index_file.write(line + "\n")

class ProcessedDataset(Dataset):
    path: str
    index: list[str]

    def __init__(self, path: str):
        self.path = path
        with open(f"{path}/index.txt") as file:
            self.index = [name.strip() for name in file]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = np.load(f"{self.path}/{self.index[index]}-text.npy", allow_pickle=False)
        mels = np.load(f"{self.path}/{self.index[index]}-mels.npy", allow_pickle=False)
        return torch.from_numpy(text), torch.from_numpy(mels)

    def __len__(self) -> int:
        return len(self.index)

def collate_samples(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch.sort(key=lambda p: p[0].shape[0], reverse=True)

    text = torch.nn.utils.rnn.pad_sequence(
        [text for text, _ in batch],
        batch_first=True,
    )
    text_lengths = torch.tensor([text.shape[0] for text, _ in batch])

    # Move channel dimension to last for padding, and swap back after
    mels = torch.nn.utils.rnn.pad_sequence(
        [mels.transpose(0, 1) for _, mels in batch],
        batch_first=True,
    )
    mels = mels.transpose(1, 2)
    mels_lengths = torch.tensor([mels.shape[1] for _, mels in batch])

    return text, text_lengths, mels, mels_lengths
