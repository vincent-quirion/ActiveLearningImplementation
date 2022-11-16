from torch import nn

loss_fn = nn.CrossEntropyLoss()

log_interval = 5


# Note : to make training faster, we only run on epoch
def train(model, train_loader, optimizer, epochs, **kwargs):
    model.train()
    for epoch in range(epochs):
        for batch_id, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                print(
                    "Train - Acquisition Function : {} Step: {}/{} - Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        kwargs.get("acquisition_function", "?"),
                        kwargs.get("step", "?"),
                        kwargs.get("max_steps", "?"),
                        epoch + 1,
                        batch_id * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_id / len(train_loader),
                        loss.item(),
                    ),
                )
