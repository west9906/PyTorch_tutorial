import torch

x = torch.tensor(1.)
y = torch.tensor(2.)

w = torch.tensor(1., requires_grad=True)

#forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

#backward pass
loss.backward()

print(w.grad)



