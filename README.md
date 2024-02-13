# A Mimumim Example for Training on MNIST

A minimum example for pytorch DDP based single node multi-GPU usage on MNIST dataset. This example includes basic DDP usage, gradient compression, and additional handling. 

## 1. Environment setup
Create a environment with pytorch in it
```sh
conda create --name torchenv python=3.9
conda activate torchenv
conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

## 2. Launching MNIST example


## 2.1. MNIST training

```sh
bash train.sh
```
You may change the available gpu with ```--gpu 0```, change gpu to the gpu you need to use. 

You can add ```--path xxx``` to change the model saving and logger path to the path you like. By default, the model and logger saving path are ```./logging/```. 

## 2.2. MNIST evaluation using existing model

```sh
bash eval.sh
```
You may change the available gpu with ```--gpu 0```, change gpu to the gpu you need to use. 

You can add ```--path xxx``` to change the model saving and logger path to the path you like. By default, the model and logger saving path are ```./logging/```. 

```--evaluate ./logging/mnist_model.pt``` gives the path of model you want to evaluate, ```--path ./logging_test/``` gives the logger directory. 

## 3. Detailed explanation:
### 3.1. Logger:
```py
logger = get_logger(os.path.join(args.path, "mnist_train.log"))
```
This set the logger output path, by default ```args.path``` is ```./logging/```

### 3.2. Model Architecture:
```py
model = eval(args.model + "()")
```
The we set the model architecture through command line argument ```--model LeNet``` or ```--model MLP```. LeNet and MLP architecture defination can be found in code. 

### 3.3. Optimizer:
```py
optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
```
We use SGD optimizer with proper momentum and weight decay. By default, the momentum is 0.9 and weight decay is 5e-4 (0.0005). 

### 3.4. LR scheduler:

```py
# Cosine Annealing Learning Rate Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
```
We use cosine annealing LR decay, which decays the lr from large number to a small number according to the cosine function. 

### 3.5. Evaluation:
```py
if args.evaluate:
    load_and_test(args.evaluate, model, device, test_loader, logger)
    exit()
```

If the ```evaluate``` command line input is given, then we directly load the corresponding model and test the model without training. 

### 3.6. Training and Logging:
```py
best_acc = 0.0
for epoch in range(1, args.epochs + 1):
    train_acc = train(args.log_interval, model, device, train_loader, optimizer, epoch, logger)
    test_acc = test(model, device, test_loader, logger)
    scheduler.step()

    if test_acc > best_acc:
        best_acc = test_acc
        logger.info("Save model with best accuracy: {:.4f}%".format(best_acc))
        torch.save(model.state_dict(), os.path.join(args.path, "mnist_model.pt"))
    logger.info("Current best test accuracy: {:.4f}%".format(best_acc))
logger.info('Final best Test Accuracy: {:.4f}%'.format(best_acc))
```
For each epoch, we train the model and test the model. Then, we use ```scheduler.step()``` to decay the LR according to CosineAnnealingLR scheduler. At last, we determine if the current epoch's test acc is higher than best test accuracy, if yes, we save the model with current epoch's acc, and set the best test acc to current epoch acc. If not, than we do nothing and continue to next epoch. 
