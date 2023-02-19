import torch


class Trainer():
    def __init__(self, lr_scheduler, patience=5, save_path='best_model.pth', best_val_loss=float('inf')):
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = best_val_loss
        self.counter = 0
        self.min_delta = 1e-3
        self.stop = False

    def __call__(self, current_valid_loss, model):
        if self.best_val_loss - current_valid_loss > self.min_delta:
            print(f'Validation loss improved from {self.best_val_loss} to {current_valid_loss}!')
            self.best_val_loss = current_valid_loss
            self.counter = 0

            print('Saving best model ...')
            torch.save(model.state_dict(), self.save_path)

        else:
            self.counter += 1
            print(
                f'Validation loss did not improve from {self.best_val_loss}! Counter {self.counter} of {self.patience}.')
            if self.counter < self.patience:
                self.lr_scheduler.step(current_valid_loss)

            else:
                self.stop = True
