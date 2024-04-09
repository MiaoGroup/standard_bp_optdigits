import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 定义MSE损失函数及其导数
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def mse_loss_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred) / y_true.shape[0]

def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

def limit_precision(weights, bitlength):
    uplimit = 2 ** (bitlength - 1)
    # boundry = np.max(np.abs(weights))
    boundry = 6
    weights [weights > boundry] = boundry
    weights [weights < -boundry] = -boundry
    tempdata = np.round (weights / boundry  * uplimit )
    return tempdata / uplimit * boundry

def add_noise(weights, bitlength):
    uplimit = 2 ** (bitlength+1)
    # boundry = np.max(np.abs(weights))
    boundry = 6
    return weights + np.random.uniform(-boundry/uplimit, boundry/uplimit, weights.shape)

Trainset = np.loadtxt('optdigits.tra.csv', delimiter = ',')
Testset = np.loadtxt('optdigits.tes.csv', delimiter = ',')
Train_x = Trainset[:, :64]
Train_y = Trainset[:, 64]
test_x = Testset[:, :64]
test_y = Testset[:, 64]

test_y = test_y.astype(int)
Train_y = Train_y.astype(int)

print('Train_x shape =', Train_x.shape)
print('Test_x shape =',test_x.shape)

X_train = Train_x / 16 # same operation with IPAL demonstration
# print(Train_x.shape)
# print(X_train.shape)
# print(Train_x)
# print(X_train)

X_test = test_x/16
y_train = np.eye(10)[Train_y]
y_test= np.eye(10)[test_y]

# print(X_train.shape)
# print(X_test.shape)
# print(y_train)
# print(X_test.shape)

input_size = X_train.shape[1]
hidden_size = 32 # same with IPAL demonstration
output_size = 10 # same with IPAL demonstration

np.random.seed(42)
w1 = np.random.normal(0, 0.5, (input_size, hidden_size))
b1 = np.zeros(hidden_size)
w2 = np.random.normal(0, 0.5, (hidden_size, output_size))
b2 = np.zeros(output_size)
print(w1.dtype)
print(w2.dtype)
print(b1.dtype)
print(b2.dtype)
# 训练参数
learning_rate = 2
epochs = 200  # same with IPAL demonstration
batch_size = 40  # same with IPAL demonstration

# 训练过程
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):

       # 计算训练集损失和精度
    z1_train = np.dot(X_train, w1) + b1
    a1_train = sigmoid(z1_train)
    z2_train = np.dot(a1_train, w2) + b2
    a2_train = sigmoid(z2_train)
    train_loss = mse_loss(y_train, a2_train)
    train_accuracy = accuracy(y_train, a2_train)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 计算测试集精度
    z1_test = np.dot(X_test, w1) + b1
    a1_test = sigmoid(z1_test)
    z2_test = np.dot(a1_test, w2) + b2
    a2_test = sigmoid(z2_test)
    test_accuracy = accuracy(y_test, a2_test)
    test_accuracies.append(test_accuracy)
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        
        # 前向传播
        z1 = np.dot(X_batch, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        
        # 反向传播
        dz2 = mse_loss_derivative(y_batch, a2) * sigmoid_derivative(a2)
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        dz1 = np.dot(dz2, w2.T) * sigmoid_derivative(a1)
        dw1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0)

       # 更新权重和偏置
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 = limit_precision(w1, 9)
        w2 = limit_precision(w2, 9)
        b1 = limit_precision(b1, 9)
        b2 = limit_precision(b2, 9)
        # print (np.random.uniform(-6/64, 6/64, w1.shape))
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 可视化训练损失、训练精度和测试精度
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', linestyle='--')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

random_numbers = np.random.randint(1, 1797, size=10)
randomSelectedX = X_test[random_numbers, :].shape
randomSelectedy = test_y[random_numbers]
fig, axs = plt.subplots(2, 10)
ii = 0
for index in random_numbers:
    axs[0, ii].imshow( X_test[index , :].reshape(8,8))
    z1 = np.dot(X_test[index , :], w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    axs[1, ii].plot( a2 )
    ii = ii + 1
plt.show()

