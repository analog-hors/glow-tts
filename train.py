import torch, torch.nn.functional as F, sys, time, lzma
from torch.utils.data import DataLoader, RandomSampler

from glow_tts.model import GlowTTS
from glow_tts.hparams import Hyperparameters
from glow_tts.loss import mle_loss
from dataset import ProcessedDataset, collate_samples, TOKENS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENABLE_AMP = True
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 5000

if __name__ == "__main__":
    dataset = ProcessedDataset("datasets/ljspeech-processed/")
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=RandomSampler(dataset, replacement=True, num_samples=1_000_000_000),
        pin_memory=True,
        num_workers=1,
        collate_fn=collate_samples,
    )

    hparams = Hyperparameters(num_symbols=len(TOKENS))
    model = GlowTTS(hparams).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-6)
    scaler = torch.GradScaler(enabled=ENABLE_AMP)
    batches = 0

    if len(sys.argv) == 2:
        checkpoint = torch.load(sys.argv[1])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        batches = checkpoint["batches"]

    model.train()
    running_start = time.time()
    running_loss = 0.0
    for text, text_lengths, mels, mels_lengths in dataloader:
        text = text.to(DEVICE, non_blocking=True)
        text_lengths = text_lengths.to(DEVICE, non_blocking=True)
        mels = mels.to(DEVICE, non_blocking=True)
        mels_lengths = mels_lengths.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast("cuda", enabled=ENABLE_AMP):
            result = model(text, text_lengths, mels, mels_lengths)

            mels_loss = mle_loss(
                result.latent,
                result.mean,
                result.log_stdev,
                result.logdet_sum,
                result.mels_mask,
            )
            
            text_mask_bool = result.text_mask != 0.0
            log_duration_target = torch.log(torch.sum(result.attn_weights, -1) + 1e-8)
            duration_loss = F.mse_loss(
                result.log_duration[text_mask_bool],
                log_duration_target[text_mask_bool],
            )

            loss = mels_loss + duration_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        batches += 1

        if batches % LOG_INTERVAL == 0:
            avg_running_loss = running_loss / LOG_INTERVAL
            batches_per_sec = LOG_INTERVAL / (time.time() - running_start)
            print(f"[{batches}] loss: {avg_running_loss}, {batches_per_sec:.2f} batches/sec", flush=True)

            running_start = time.time()
            running_loss = 0.0

        if batches % CHECKPOINT_INTERVAL == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "batches": batches,
            }, f"checkpoints/{batches:06}-model.pth")
