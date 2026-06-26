import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


device_ids = [0, 1]

"""路径参数及超参设置"""
CONFIG = {
    "func_bert_path": "/data/prqu/prqu_files/pretrained_models/Unixcoder",
    "max_func_length": 512,
    "num_classes": 2,
    "batch_size": 16,
    "lr": 1e-5,
    "epochs": 50,
    "device": torch.device(f"cuda:{device_ids[1]}" if torch.cuda.is_available() else "cpu"),
    "log_dir": "/data/prqu/prqu_files/dir_logs_test_2",
    "log_file": "/data/prqu/prqu_files/save_file/training_logs_Unixcoder_func_only.txt",
    "save_file": "/data/prqu/prqu_files/save_file",
    "model_dir": "/data/prqu/prqu_files/save_file/models_func_only_Unixcoder",
    "max_saved_models": 1,
}


"""数据集处理（仅函数级）"""
class CodeDataset(Dataset):
    def __init__(self, df, max_func_length=512):
        self.df = df
        self.max_func_length = max_func_length
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["func_bert_path"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        func_before = str(self.df.iloc[idx]["func_before"])
        func_label = int(self.df.iloc[idx]["fun_label"])

        func_encoding = self.tokenizer(
            func_before,
            max_length=self.max_func_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )

        return {
            "func_input_ids": func_encoding["input_ids"].squeeze(0),
            "func_attention_mask": func_encoding["attention_mask"].squeeze(0),
            "func_label": torch.tensor(func_label, dtype=torch.long),
        }


def custom_collate(batch):
    return {
        "func_input_ids": torch.stack([sample["func_input_ids"] for sample in batch]),
        "func_attention_mask": torch.stack([sample["func_attention_mask"] for sample in batch]),
        "func_label": torch.stack([sample["func_label"] for sample in batch]),
    }


"""模型类（仅函数级）"""
class VulDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unixcoder = AutoModel.from_pretrained(CONFIG["func_bert_path"])

        for param in self.unixcoder.parameters():
            param.requires_grad = False

        if hasattr(self.unixcoder, "encoder") and hasattr(self.unixcoder.encoder, "layer"):
            for param in self.unixcoder.encoder.layer[-6:].parameters():
                param.requires_grad = True
        else:
            # 回退策略：如果模型结构非标准RoBERTa，至少保持可训练。
            for param in self.unixcoder.parameters():
                param.requires_grad = True

        self.func_classifier = nn.Sequential(
            nn.Linear(self.unixcoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, CONFIG["num_classes"]),
        )

    def forward(self, func_inputs):
        encoder_outputs = self.unixcoder(
            input_ids=func_inputs["input_ids"],
            attention_mask=func_inputs["attention_mask"],
        )

        last_hidden = encoder_outputs.last_hidden_state
        attention_mask = func_inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())

        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
        token_count = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_features = sum_hidden / torch.clamp(token_count, min=1)

        func_logits = self.func_classifier(pooled_features)
        return {"func_logits": func_logits}


"""计算函数级指标"""
def calculate_func_metrics(preds, labels):
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    metrics = {
        "acc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    if len(labels) == 0:
        return metrics

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    total = tp + fp + tn + fn
    metrics["acc"] = (tp + tn) / total if total else 0.0
    metrics["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["f1"] = (
        2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        if (metrics["precision"] + metrics["recall"])
        else 0.0
    )
    return metrics


def write_logfile(phase, epoch, func_metrics, loss):
    log_entry = (
        f"{phase}:Epoch {epoch:03d}:\n"
        f"\t[Func] Acc: {func_metrics['acc']:.4f} | F1: {func_metrics['f1']:.4f} | "
        f"Pre: {func_metrics['precision']:.4f} | Rec: {func_metrics['recall']:.4f}\n"
        f"\tLoss: {loss:.4f}\n"
    )
    with open(CONFIG["log_file"], "a", encoding="utf-8") as f:
        f.write(log_entry)


def print_metrics(phase, loss, func_metrics):
    print(
        f"{phase} Loss: {loss:.4f} | Func_Acc: {func_metrics['acc']:.4f} | "
        f"Func_Pre: {func_metrics['precision']:.4f} | Func_Rec: {func_metrics['recall']:.4f} | "
        f"Func_F1: {func_metrics['f1']:.4f}"
    )


def save_top_models(current_f1, epoch, model):
    global top_models

    if len(top_models) < CONFIG["max_saved_models"] or current_f1 > top_models[-1]["f1"]:
        model_name = f"epoch_{epoch:03d}__f1_{current_f1:.4f}.pth"
        model_path = os.path.join(CONFIG["model_dir"], model_name)

        torch.save(model.state_dict(), model_path)
        top_models.append({"f1": current_f1, "epoch": epoch, "path": model_path})
        top_models.sort(key=lambda x: -x["f1"])

        if len(top_models) > CONFIG["max_saved_models"]:
            removed = top_models.pop()
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    current_step = epoch * len(dataloader)

    total_loss = 0.0
    func_preds, func_labels = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        func_inputs = {
            "input_ids": batch["func_input_ids"].to(device),
            "attention_mask": batch["func_attention_mask"].to(device),
        }
        labels = batch["func_label"].to(device)

        outputs = model(func_inputs)
        loss = criterion(outputs["func_logits"], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], current_step + batch_idx)
        writer.add_scalar("Train/Batch_Loss", loss.item(), current_step + batch_idx)

        batch_preds = torch.argmax(outputs["func_logits"], dim=-1).cpu().numpy()
        batch_labels = labels.cpu().numpy()
        func_preds.extend(batch_preds)
        func_labels.extend(batch_labels)

        # if batch_idx >= 10:
        #     break

    pbar.close()

    avg_loss = total_loss / len(dataloader)
    func_metrics = calculate_func_metrics(func_preds, func_labels)

    writer.add_scalars(
        "Epoch/Train_Func_Metrics",
        {
            "Func_F1": func_metrics["f1"],
            "Func_Acc": func_metrics["acc"],
            "Func_Precision": func_metrics["precision"],
            "Func_Recall": func_metrics["recall"],
        },
        epoch,
    )

    return avg_loss, func_metrics


def evaluate(model, dataloader, device, epoch, writer, mode="val"):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    func_preds, func_labels = [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [{mode.upper()}]", unit="batch")

        for batch in pbar:
            func_inputs = {
                "input_ids": batch["func_input_ids"].to(device),
                "attention_mask": batch["func_attention_mask"].to(device),
            }
            labels = batch["func_label"].to(device)

            outputs = model(func_inputs)
            loss = criterion(outputs["func_logits"], labels)

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            batch_preds = torch.argmax(outputs["func_logits"], dim=-1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            func_preds.extend(batch_preds)
            func_labels.extend(batch_labels)

        pbar.close()

    avg_loss = total_loss / len(dataloader)
    func_metrics = calculate_func_metrics(func_preds, func_labels)

    writer.add_scalars(
        f"Epoch/{mode}_Func_Metrics",
        {
            "Func_F1": func_metrics["f1"],
            "Func_Acc": func_metrics["acc"],
            "Func_Precision": func_metrics["precision"],
            "Func_Recall": func_metrics["recall"],
        },
        epoch,
    )

    return avg_loss, func_metrics


def test(model, test_loader, device, writer):
    if len(top_models) > 0:
        best_model_path = top_models[0]["path"]
        print(f"加载最佳模型: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location="cpu", weights_only=True))
        model.to(device)
    else:
        print("警告: 没有已保存的最优模型，使用当前模型直接测试")

    test_loss, test_func_metrics = evaluate(model, test_loader, device, 0, writer, mode="test")
    write_logfile("Test_Metrics", 0, func_metrics=test_func_metrics, loss=test_loss)
    print_metrics("Test", test_loss, test_func_metrics)


if __name__ == "__main__":

    writer = SummaryWriter(CONFIG["log_dir"])
    top_models = []

    train_df = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/train_balanced.csv")
    val_df = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/val_balanced.csv")

    train_dataset = CodeDataset(train_df, max_func_length=CONFIG["max_func_length"])
    val_dataset = CodeDataset(val_df, max_func_length=CONFIG["max_func_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
    )

    model = VulDetectionModel().to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.1)

    num_training_steps = CONFIG["epochs"] * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    for epoch in range(CONFIG["epochs"]):
        print("\n==========开始训练阶段===========\n")
        train_loss, train_func_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            CONFIG["device"],
            epoch,
            writer,
        )
        write_logfile("Train_Metrics", epoch, func_metrics=train_func_metrics, loss=train_loss)

        print("\n==========开始验证阶段===========\n")
        val_loss, val_func_metrics = evaluate(
            model,
            val_loader,
            CONFIG["device"],
            epoch,
            writer,
            mode="val",
        )
        write_logfile("Val_Metrics", epoch, func_metrics=val_func_metrics, loss=val_loss)

        writer.add_scalars("Loss/Train_vs_Val", {"Train": train_loss, "Validation": val_loss}, epoch)

        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print_metrics("Train", train_loss, train_func_metrics)
        print_metrics("Val", val_loss, val_func_metrics)

        save_top_models(val_func_metrics["f1"], epoch, model)
        print("\n当前Top模型列表:")
        for i, m in enumerate(top_models, 1):
            print(f"第{i}名 | Epoch {m['epoch']} | F1: {m['f1']:.4f}")

    print("\n" + "=" * 60)
    print("训练完成! 开始测试阶段...")
    print("=" * 60)

    test_df = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/test_balanced.csv")
    test_dataset = CodeDataset(test_df, max_func_length=CONFIG["max_func_length"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
    )

    test(model, test_loader, CONFIG["device"], writer)
    writer.close()