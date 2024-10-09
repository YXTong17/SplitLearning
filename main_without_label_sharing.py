import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ClientModel, ServerModel


class Client:
    def __init__(self, input_size, hidden_size, learning_rate):
        self.model = ClientModel(input_size, hidden_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.data = None
        self.labels = None

    def send_activations(self, activations, server):
        server.activations = activations.detach()
        server.activations.requires_grad_()

    def send_output_gradients(self, output_gradients, server):
        server.output_gradients = output_gradients


class Server:
    def __init__(self, hidden_size, output_size, learning_rate):
        self.model = ServerModel(hidden_size, output_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def send_output(self, output, client):
        client.output = output.detach()
        client.output.requires_grad_()

    def send_activations_gradients(self, activations_gradients, client):
        client.activations_gradients = activations_gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--frozen_server", type=bool, default=True)
    args = parser.parse_args()

    client = Client(args.input_size, args.hidden_size, args.learning_rate)
    client.data = torch.randn(args.num_samples, args.input_size)
    client.labels = torch.randint(0, args.num_classes, (args.num_samples,))

    server = Server(args.hidden_size, args.num_classes, args.learning_rate)
    if args.frozen_server:
        for param in server.model.parameters():
            param.requires_grad = False

    for epoch in range(args.epochs):
        client.optimizer.zero_grad()
        server.optimizer.zero_grad()

        """ 客户端前向传播，生成激活值，发送给服务器 """
        client.activations = client.model(client.data)
        client.send_activations(client.activations, server)

        """ 服务器继续前向传播，将结果发送给客户端 """
        server.output = server.model(server.activations)
        server.send_output(server.output, client)

        """ 客户端计算损失，进行第一次反向传播，并将梯度发送到服务器 """
        loss = client.criterion(client.output, client.labels)
        output_gradients = torch.autograd.grad(loss, client.output)
        client.send_output_gradients(output_gradients, server)

        """ 服务器反向传播，结果发送给客户端，更新模型 """
        server.output.backward(server.output_gradients)
        server.send_activations_gradients(server.activations.grad, client)
        server.optimizer.step()

        """ 客户端进行第二次反向传播，更新模型 """
        client.activations.backward(client.activations_gradients)
        client.optimizer.step()

        # 测试准确率
        with torch.no_grad():
            activations = client.model(client.data).detach()
            output = server.model(activations)
            _, predicted = torch.max(output, 1)
            correct = (predicted == client.labels).sum().item()
            accuracy = correct / client.labels.size(0)
            print(
                f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%"
            )


if __name__ == "__main__":
    main()
