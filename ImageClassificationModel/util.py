from torch import save, load
from os import listdir


def save_and_log(
    model,
    id,
    epoch,
    batch,
    metrics,
    models_dir="ImageClassificationModel/models",
    log_dir="ImageClassificationModel/log.txt",
):
    save(model, f"{models_dir}/{id}.pt")
    with open(log_dir, "a") as f:
        f.write(
            f"Model {id}\n\tEpoch: {epoch}\n\tBatch: {batch}\n\tLoss: {metrics[0]}\n\tAccuracy: {metrics[1]}\n\tPrecision: {metrics[2]}\n\tRecall: {metrics[3]}\n\n"
        )


def load(id, models_dir="ImageClassificationModel/models"):
    model = load(f"{models_dir}/{id}.pt")
    return model


def newest_model(models_dir):
    max = -1
    for dir in listdir(models_dir):
        if int(dir[:-3]) > max:
            max = int(dir[:-3])
    return max


def validate(model, loader, loss_fn):
    n_batches = loader.n_val_batches
    for batch in range(n_batches):
        x, yhat = loader.next_val_batch()
        y = model(x)
        loss = loss_fn(y, yhat)
        y = y.detach().cpu().squeeze(1).numpy()
        yhat = yhat.detach().cpu().squeeze(1).numpy()
        outcomes = calculate_outcomes(yhat, y)
        accuracy = accuracy_score(yhat, y)
        precision = precision_score(outcomes)
        recall = recall_score(outcomes)
        return loss.item(), accuracy, precision, recall, outcomes


def calculate_outcomes(yhat, y):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(yhat.shape[0]):
        if yhat[i] == 1 and round(y[i]) == 1:
            TP += 1
        elif yhat[i] == 0 and round(y[i]) == 1:
            FP += 1
        elif yhat[i] == 0 and round(y[i]) == 0:
            TN += 1
        elif yhat[i] == 1 and round(y[i]) == 0:
            FN += 1

    return (TP, FP, TN, FN)


def accuracy_score(yhat, y):
    correct = 0
    for i in range(yhat.shape[0]):
        if yhat[i] == round(y[i]):
            correct += 1
    return correct / yhat.shape[0]


def precision_score(outcomes):
    TP, FP, TN, FN = outcomes
    return TP / (TP + FP)


def recall_score(outcomes):
    TP, FP, TN, FN = outcomes
    return TP / (TP + FN)

