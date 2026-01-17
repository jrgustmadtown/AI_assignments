import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms # type: ignore



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)
    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform= transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])

    train_set = datasets.FashionMNIST(root="./data", train=True, transform=custom_transform, download=True)
    test_set = datasets.FashionMNIST(root="./data", train=False, transform=custom_transform, download=True)

    if training==False:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
        return loader
    
    loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model



def build_deeper_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    model.train()
    optimer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for t in range(T):
        correct=0
        sumloss=0
        sumsamples=0
        for img, lbl in train_loader:
            optimer.zero_grad()
            outputs=model(img)
            loss=criterion(outputs,lbl)
            loss.backward()
            optimer.step()
            sumloss+=loss.item()*img.size(0)
            x, predict=torch.max(outputs,1)
            correct+=(predict==lbl).sum().item()
            sumsamples+=lbl.size(0)
        acc=100*correct/sumsamples
        avgloss=sumloss/sumsamples
        print(f"Train Epoch: {t} Accuracy: {correct}/{sumsamples}({acc:.2f}%) Loss: {avgloss:.3f}")


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    correct = 0
    sumloss=0
    sumsamples=0

    with torch.no_grad():
        for img, lbl in test_loader:
            outputs = model(img) 
            loss = criterion(outputs, lbl)  
            sumloss += loss.item() * img.size(0) 
            _, predicted = torch.max(outputs, 1) 
            correct += (predicted == lbl).sum().item()  
            sumsamples += lbl.size(0) 
    
    accuracy = 100.0 * correct / sumsamples  
    avg_loss = sumloss / sumsamples  
    
    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")



def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    model.eval()
    with torch.no_grad():
        log = model(test_images[index].unsqueeze(0)) 
        prob = F.softmax(log, dim=1) 
        top, top_lbl = torch.topk(prob, 3, dim=1)  
    
    for i in range(3):
        label = classes[top_lbl[0][i].item()]
        probability = top[0][i].item() * 100
        print(f"{label}: {probability:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
