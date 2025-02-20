import torch
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
import wandb
import math
from noodlepy.utils.embeddingviewer import EmbeddingViewer


def train_model(model, train_dataloader, training_cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = NegativeCosineSimilarity().to(device)
    # criterion = NTXentLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=training_cfg.learning_rate, momentum=0.9, weight_decay=1e-4)

        # training
    print("Starting Training")
    for epoch in range(training_cfg.epochs):
        avg_loss = 0.0
        avg_output_std = 0.0
        #total_loss = 0.0
        for i, (x0, x1, labels, raman_shift) in enumerate(train_dataloader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0, p0 = model(x0)
            z1, p1 = model(x1)
            loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
            #total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # calculate the per-dimension standard deviation of the outputs
            # we can use this later to check whether the embeddings are collapsing
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)
            output_std = torch.std(output, 0)
            output_std = output_std.mean()
            # use moving averages to track the loss and standard deviation
            w = 0.9
            avg_loss = w * avg_loss + (1 - w) * loss.item()
            avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

        # log the loss of every epoch
        wandb.log({"loss": avg_loss, "output_std": avg_output_std, "epoch": epoch})
        print(f"Epoch {epoch+1}/{training_cfg.epochs}, Loss: {avg_loss:.4f}, Output Std: {avg_output_std:.4f}")
        # save the model every 10 epochs
    print("Finished Training")
    return model

def test_model(model, test_dataloader, label_name_for_color='staging', map_for_color = None, exp_name = "test"):
    print("Visualizing inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = []
    label_dict_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, _, labels, raman_shift) in enumerate(test_dataloader):
            x = x.to(device)
            y = model.backbone(x).flatten(start_dim=1)
            embeddings.append(y)
            # add x to the list of labels as intensity
            labels['intensity'] = x
            labels['raman_shift'] = raman_shift
            label_dict_list.append(labels)

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    viewer = EmbeddingViewer(embeddings, label_dict_list, map=map_for_color, exp_name=exp_name)
    viewer.tsne2d(label_name_for_color=label_name_for_color)
    viewer.save_files_for_tf_embedding_projector()
