import torch
import torch.nn as nn
import torch.optim as optim


class PerceptronNN(nn.Module):
    def __init__(self, input_size, num_layer, hidden_size):
        super(PerceptronNN, self).__init__()

        assert (
            len(hidden_size) >= num_layer
        ), "hidden_size must have at least num_layer elements"

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))

        for i in range(1, num_layer):
            layers.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))

        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size[num_layer - 1], 10)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        with open("training_log.txt", "a") as f:

            def print_and_log(message):
                print(message)
                f.write(message + "\n")
                f.flush()  # Ensure that the log file is updated immediately

            print_and_log("===== Start of Training =====")
            previous_loss = float("inf")
            patience = 5
            for epoch in range(0, num_epochs + 1):
                running_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()

                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                average_loss = running_loss / len(train_loader)

                if epoch % (num_epochs // 10) == 0:
                    print_and_log(f"Epoch {epoch}, Average Loss: {average_loss:.4f}")

                if average_loss > previous_loss:
                    patience -= 1
                else:
                    patience = 5

                if patience == 0:
                    print_and_log(
                        f"Stopping at epoch {epoch} with average loss of {average_loss:.4f} due to consistent loss increase."
                    )
                    break

                if abs(previous_loss - average_loss) < 1e-4:
                    print_and_log(
                        f"Stopping at epoch {epoch} with average loss of {average_loss:.4f} due to loss change less than 1e-4."
                    )
                    break

                previous_loss = average_loss

            print_and_log("====== End of Training ======\n")

    def validate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in test_loader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total  # Percentage of correctly predicted labels
        return (accuracy, all_labels, all_predictions)
