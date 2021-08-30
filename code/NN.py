import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        # apply_softmax (bool): a flag for softmax activation should be false if using Cross Entropy Loss
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

def training_loop(X_train, y_train, X_test, learning_rate=0.02, num_epochs=10, batch_size=26):
    device = 'cpu'
    classifier = Classifier(input_dim=3, hidden_dim=2, output_dim=2)
    classifier = classifier.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # split data into batches
    X_train, y_train,  X_test =  torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test)
    X_train_batches = torch.split(X_train, batch_size)
    y_train_batches = torch.split(y_train, batch_size)

    for epoch in range(num_epochs):
        running_loss = 0.0
        classifier.train()
        batch_index = 0

        for local_batch, local_labels in zip(X_train_batches, y_train_batches):
            # clear gradients
            optimizer.zero_grad()
            # compute output
            y_pred = classifier(local_batch.float())
            # compute loss
            loss = loss_func(y_pred, local_labels)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # produce gradients
            loss.backward()
            # backpropogation
            optimizer.step()
            batch_index += 1
        print("Epoch {}: ".format(epoch + 1))
        print("  Train Loss: {}".format(running_loss))

    # get predictions
    classifier.eval()
    with torch.set_grad_enabled(False):
        predictions = X_test.float().detach().numpy()
        predictions = np.argmax(predictions, axis=1)

    return predictions
