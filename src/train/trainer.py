import os
import time
from dataclasses import dataclass

import torch
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler

from src.accuracy_eval import AccuracyEval
from src.log_utils import log_title


@dataclass
class CheckpointConfig:
    save_interval: int
    checkpoint_dir: str
    latest_checkpoint_name: str
    best_checkpoint_name: str
    interval_checkpoint_name: str


class ModelTrainer:
    def __init__(
            self,
            model,
            dataset_train,
            dataset_evaluate,
            criterion,
            checkpoint_config: CheckpointConfig,
            resume_training=False,
            cuda=True,
            loss_log_steps_interval=50,
            num_classes=2,
    ):
        self.model = model if not cuda else torch.nn.DataParallel(model).cuda()
        self.dataset_train = dataset_train
        self.dataset_evaluate = dataset_evaluate
        self.criterion = criterion
        self.checkpoint_config = checkpoint_config

        self.resume_training = resume_training
        self.cuda = cuda

        self.optimizer = Adam(self.model.parameters(), 5e-4, (0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        self.best_accuracy = 0
        self.start_epoch = 1

        self.automated_log_file = f"{self.checkpoint_config.checkpoint_dir}/train_log.txt"
        self.model_txt_file = f"{self.checkpoint_config.checkpoint_dir}/model.txt"
        self.loss_log_steps_interval = loss_log_steps_interval
        self.num_classes = num_classes

    def train(self, epochs):
        log_title("START TRAINING")

        # Set up log files
        if not os.path.exists(self.checkpoint_config.checkpoint_dir):
            os.makedirs(self.checkpoint_config.checkpoint_dir)

        self.init_logs()

        if self.resume_training:
            self.load_checkpoint()

        lr_lambda = lambda e: pow((1 - ((e - 1) / epochs)), 0.9)
        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        for epoch in range(self.start_epoch, epochs + 1):
            log_title(f"TRAINING EPOCH {epoch}")

            epoch_loss = []
            time_train = []

            train_iou_accuracy = AccuracyEval()
            learning_rate = 0
            for param_group in self.optimizer.param_groups:
                print(f"Learning Rate: {param_group.get('lr')}")
                learning_rate = float(param_group.get("lr"))

            self.model.train()
            for step, (images, labels) in enumerate(self.dataset_train):
                start_time = time.time()

                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                inputs = Variable(images)
                targets = Variable(labels)
                outputs = self.model(inputs)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, targets[:, 0])
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.data.item())
                time_train.append(time.time() - start_time)

                train_iou_accuracy.add_batch(outputs, labels)

                if self.loss_log_steps_interval > 0 and step % self.loss_log_steps_interval == 0:
                    avg_loss = sum(epoch_loss) / len(epoch_loss)
                    print(f"[TRAIN] Loss: {avg_loss:04}, Epoch: {epoch}, Step: {step}")

            average_epoch_train_loss = sum(epoch_loss) / len(epoch_loss)

            train_accuracy, _ = train_iou_accuracy()

            log_title(f"EVALUATING EPOCH {epoch}")

            epoch_loss_evaluate = []
            time_evaluate = []

            evaluate_iou_accuracy = AccuracyEval()

            self.model.eval()
            for step, (images, labels) in enumerate(self.dataset_evaluate):
                start_time = time.time()

                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                inputs = Variable(images)
                targets = Variable(labels)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets[:, 0])
                epoch_loss_evaluate.append(loss.data.item())
                time_evaluate.append(time.time() - start_time)

                evaluate_iou_accuracy.add_batch(outputs, labels)

                if self.loss_log_steps_interval > 0 and step % self.loss_log_steps_interval == 0:
                    avg_loss = sum(epoch_loss_evaluate) / len(epoch_loss_evaluate)
                    print(f"[EVALUATE] Loss: {avg_loss:04}, Epoch: {epoch}, Step: {step}")

            average_epoch_eval_loss = sum(epoch_loss_evaluate) / len(epoch_loss_evaluate)

            eval_accuracy, _ = evaluate_iou_accuracy()

            current_accuracy = train_accuracy
            is_best = current_accuracy > self.best_accuracy
            self.best_accuracy = max(current_accuracy, self.best_accuracy)

            self.save_checkpoint(epoch, is_best)

            scheduler.step()

            self.log_epoch(epoch, average_epoch_train_loss, average_epoch_eval_loss, train_accuracy, eval_accuracy, learning_rate)

        return self.model

    def init_logs(self):
        if os.path.exists(self.automated_log_file):
            return

        with open(self.automated_log_file, "a") as log_file:
            log_file.write("Epoch\t\tTrain Loss\t\t\t\tEval Loss\t\t\t\tTrain IOU\t\t\t\tValidate IOU\t\t\t\tLR")

        with open(self.model_txt_file, "w") as model_txt_file:
            model_txt_file.write(str(self.model))

    def log_epoch(self, epoch, avg_loss_train, avg_loss_eval, iou_train, iou_eval, lr):
        with open(self.automated_log_file, "a") as log_file:
            log_file.write("\n")
            log_file.write(f"{epoch:04}\t\t{avg_loss_train:04.16f}\t\t{avg_loss_eval:04.16f}\t\t{iou_train:04.16f}\t\t{iou_eval:04.16f}\t\t{lr:08.16f}")

    def load_checkpoint(self):
        checkpoint_config = self.checkpoint_config
        checkpoint_file = os.path.join(checkpoint_config.checkpoint_dir, checkpoint_config.latest_checkpoint_name)
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        self.start_epoch = checkpoint.get("epoch")
        self.model.load_state_dict(checkpoint.get("model_state_dict"))
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))
        self.best_accuracy = checkpoint.get("best_accuracy")
        print(f"Resuming training at epoch {self.start_epoch}")

    def save_checkpoint(self, epoch, is_best):
        checkpoint_config = self.checkpoint_config

        latest_checkpoint = os.path.join(checkpoint_config.checkpoint_dir, checkpoint_config.latest_checkpoint_name)
        best_checkpoint = os.path.join(checkpoint_config.checkpoint_dir, checkpoint_config.best_checkpoint_name)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy,
        }

        torch.save(state, latest_checkpoint)
        if is_best:
            torch.save(state, best_checkpoint)

        save_epoch_interval = checkpoint_config.save_interval
        if save_epoch_interval > 0 and epoch % save_epoch_interval == 0:
            interval_checkpoint = os.path.join(
                checkpoint_config.checkpoint_dir,
                checkpoint_config.interval_checkpoint_name.format(epoch)
            )
            torch.save(state, interval_checkpoint)
