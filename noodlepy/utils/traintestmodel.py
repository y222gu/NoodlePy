import torch
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
import wandb
import math
from noodlepy.utils.embeddingviewer import EmbeddingViewer


def train_model(model, train_dataloader, training_cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = NegativeCosineSimilarity()
    # criterion = NTXentLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=training_cfg.learning_rate)

        # training
    print("Starting Training")
    for epoch in range(training_cfg.epochs):
        avg_loss = 0.0
        avg_output_std = 0.0
        #total_loss = 0.0
        for i, (x0, x1, labels) in enumerate(train_dataloader):
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

        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        collapse_level = max(0.0, 1 - math.sqrt(64) * avg_output_std)
        wandb.log({'epoch': epoch+1, 'loss': avg_loss, 'collapse_level': collapse_level})
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}, collapse_level: {collapse_level:.5f}")
    print("Finished Training")
    return model

def test_model(model, test_dataloader, label_name_for_color='staging', label_name_for_marker='sample_type', map_for_color_and_marker = None, exp_name = "test"):
    print("Visualizing inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = []
    label_dict_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, _, labels) in enumerate(test_dataloader):
            x = x.to(device)
            y = model.backbone(x).flatten(start_dim=1)
            embeddings.append(y)
            label_dict_list.append(labels)

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    viewer = EmbeddingViewer(embeddings, label_dict_list, map=map_for_color_and_marker, exp_name=exp_name)
    viewer.tsne2d(label_name_for_color=label_name_for_color, label_name_for_marker=label_name_for_marker)
    viewer.save_files_for_tf_embedding_projector()
