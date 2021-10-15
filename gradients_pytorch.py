import torch

# f = w * x

# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0., dtype=torch.float32, requires_grad=True)


#model prediction
def forward(x):
    return w * x


#loss
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'Prediction before trainging: f(5)={forward(5):.3f}')

#Traingin
lr = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y,  y_pred)

    #gradients = backward
    l.backward() #dl/dw

    #update weights
    with torch.no_grad(): # w should be not a part of tracking graph
        w -= lr * w.grad

    #zero gradients
    w.grad.zero_()

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}:w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after trainginc: f(5)={forward(5):.3f}')