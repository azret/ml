import time
import torch
import torch.nn.functional as F

def pretty_logits(logits):
    logits0 = logits.view(-1)
    row = "["
    for j in range(min(len(logits0), 7)):
        row += f"{(round(logits0[j].item(), 4)):.4f}"
        if j == len(logits0) - 1:
            row += ""
        elif j > 0 and ((j + 1) % logits.shape[0]) == 0:
            row += ", "
        else:
            row += ", "
            
    row += "]"
    return row

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.transformer = torch.nn.Sequential(
            torch.nn.Linear(768, 4 * 768, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4 * 768, 768, bias=True),
            torch.nn.Sigmoid(),
        );

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x):
        x = self.transformer(x)
        return x
    
if __name__ == "__main__":
    torch.manual_seed(137)

    model = MLP()

    print(f"parameters: {model.get_num_params()}")

    lr = 1e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    batch_size = 1024

    x = torch.zeros(batch_size, 768);
    y = torch.ones(batch_size, 768);

    epochs = 100

    for epoch in range(epochs):
        start_time = time.time()
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward();
        optimizer.step()

        elapsedTicks = (time.time() - start_time)*1000;

        if (epoch == epochs - 1):
            print(pretty_logits(logits))

        print(f"{epoch}: lr = {lr}, loss={round(loss.item(), 4):.4f}, {elapsedTicks} ms")
