import torch
import torch.nn.functional as F


def generate_music(model, start_note, device, allowed_instruments, num_notes=100, temperature=1.0):
    model.eval()
    generated_sequence = start_note.clone().to(device)

    with torch.no_grad():
        for _ in range(num_notes):
            preds = model(generated_sequence)

            last_pitch_logits = preds['pitch'][:, -1, :] / temperature
            last_inst_logits = preds['instrument'][:, -1, :] / temperature
            last_vel = preds['velocity'][:, -1].unsqueeze(-1)
            last_dur = preds['duration'][:, -1].unsqueeze(-1)
            last_delta = preds['delta'][:, -1].unsqueeze(-1)

            mask = torch.ones_like(last_inst_logits, dtype=torch.bool)
            mask[:, allowed_instruments] = False
            last_inst_logits.masked_fill_(mask, -float('inf'))

            pitch_probs = F.softmax(last_pitch_logits, dim=-1)
            next_pitch = torch.multinomial(pitch_probs, num_samples=1)

            inst_probs = F.softmax(last_inst_logits, dim=-1)
            next_inst = torch.multinomial(inst_probs, num_samples=1)

            next_vel = torch.clamp(last_vel, min=0, max=127)
            next_dur = torch.clamp(last_dur, min=0.05, max=5.0)
            next_delta = torch.clamp(last_delta, min=0.0, max=4.0)

            next_note = torch.cat(
                [next_pitch.float(), next_vel, next_dur, next_inst.float(), next_delta], dim=-1).unsqueeze(1)

            generated_sequence = torch.cat(
                [generated_sequence, next_note], dim=1)

    return generated_sequence.squeeze(0)
