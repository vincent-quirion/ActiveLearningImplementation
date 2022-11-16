import torch


def eval_model(model, test_loader, **kwargs):
    model.eval()
    correct_preds = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_preds += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct_preds / len(test_loader.dataset)

    print(
        "Test - Acquisition Function : {} Step: {}/{} \tAccuracy: {:.6f}".format(
            kwargs.get("acquisition_function", "?"), kwargs.get("step", "?"), kwargs.get("max_steps", "?"), accuracy
        ),
    )

    return accuracy
