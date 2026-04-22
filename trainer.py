import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')


def train_phase(
    model, train_loader, val_loader, optimizer, num_epochs, device,
    phase_name, checkpoint_dir, weights_save_path, plot_save_path,
    min_epochs=5, patience=3
):
    """
    Handles the standard training loop for a specific phase,
    including saving checkpoints,
    early stopping based on validation loss, and plotting.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_triggered = False
    sweet_spot_epoch = 0

    print(
        f"\nStarting {phase_name} Training (Max {num_epochs} Epochs, "
        f"Min {min_epochs} Epochs)..."
    )

    for epoch in range(num_epochs):
        # ==========================================
        # 1. TRAINING LOOP
        # ==========================================
        model.train()
        total_train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            preds = model(x)

            pitch_loss = criterion_ce(
                preds['pitch'].reshape(-1, 128), y[:, :, 0].long().reshape(-1))
            inst_loss = criterion_ce(
                preds['instrument'].reshape(-1, 128), y[:, :, 3].long().reshape(-1))
            vel_loss = criterion_mse(preds['velocity'], y[:, :, 1])
            dur_loss = criterion_mse(preds['duration'], y[:, :, 2])
            delta_loss = criterion_mse(preds['delta'], y[:, :, 4])

            loss = pitch_loss + inst_loss + vel_loss + dur_loss + delta_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ==========================================
        # 2. VALIDATION LOOP
        # ==========================================
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():  # Disable gradients for faster, memory-efficient testing
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)

                pitch_loss = criterion_ce(
                    preds['pitch'].reshape(-1, 128), y[:, :, 0].long().reshape(-1))
                inst_loss = criterion_ce(
                    preds['instrument'].reshape(-1, 128), y[:, :, 3].long().reshape(-1))
                vel_loss = criterion_mse(preds['velocity'], y[:, :, 1])
                dur_loss = criterion_mse(preds['duration'], y[:, :, 2])
                delta_loss = criterion_mse(preds['delta'], y[:, :, 4])

                loss = pitch_loss + inst_loss + vel_loss + dur_loss + delta_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"--- Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {
                avg_val_loss:.4f} ---"
        )

        cp_path = os.path.join(
            checkpoint_dir, f'{phase_name.lower().replace(" ", "_")}_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), cp_path)

        # ==========================================
        # 3. EARLY STOPPING LOGIC (Now tracking Val Loss)
        # ==========================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_save_path)
            print(
                f"[*] New best validation score! Saved optimal weights to "
                f"{weights_save_path}")
        else:
            epochs_no_improve += 1
            print(f"[!] Validation loss did not improve. Patience: {
                  epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience and (epoch + 1) >= min_epochs:
            print(
                f"\nEPOT HIT! Validation loss hasn't improved for "
                f"{patience} consecutive epochs."
            )
            print(
                f"Stopping early at Epoch {epoch + 1} "
                "to prevent overfitting."
            )
            early_stop_triggered = True
            sweet_spot_epoch = epoch + 1
            break

    # ==========================================
    # 4. PLOTTING AND SUMMARY
    # ==========================================
    plt.figure(figsize=(10, 6))
    actual_epochs = len(train_losses)
    plt.plot(range(1, actual_epochs + 1), train_losses, marker='o',
             linestyle='-', label='Training Loss', color='blue')
    plt.plot(range(1, actual_epochs + 1), val_losses, marker='s',
             linestyle='-', label='Validation Loss', color='orange')

    if early_stop_triggered:
        optimal_epoch = sweet_spot_epoch - patience
        plt.axvline(x=optimal_epoch, color='r', linestyle='--',
                    label=f'Optimal Epoch ({optimal_epoch})')

    plt.title(f'{phase_name} Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_save_path)
    plt.close()

    print("\n" + "-"*50)
    if early_stop_triggered:
        optimal_epoch = sweet_spot_epoch - patience
        print(f"[SUMMARY] {phase_name} finished early.")
        print(f"The sweet spot was hit at Epoch "
              f"{optimal_epoch}, and training was halted at Epoch "
              f"{sweet_spot_epoch}.")
    else:
        print(f"[SUMMARY] {phase_name} completed all "
              f"{num_epochs} scheduled epochs without plateauing.")
    print(f"The best weights have been saved to: {weights_save_path}")
    print("-" * 50 + "\n")
