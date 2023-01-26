# Test task : You need to write a server to classify the picture then containerize it in the docker and make the server globally accessible. 
### Note : app was developed on ubuntu 20.04. Docker was tested on ubuntu and macos. On macOs it can be warnings messages .

## ⚙️ Setup & Launch
### Note : Download libtorch before build local app

``` bash

1) clone repo 
git clone https://github.com/vladimirlabynko/image_classification.git

2) go to repo directory
cd image_classification 

3) make build folder and build it 
mkdir build
cmake .. or  cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch..
make

4) After building app run app
./image classification
```

After app start you'll see next message :
```bash
== Model [../resnet18.pt] loaded!
== Label loaded! Let's try it
```

While server is running,open another terminal and send CURL requests from your local pc like this : 
```bash 

curl -X POST -d "urltoimage" http://localhost:12345
```

Wait a seconds and you'll see a predicted class like this :
```bash
Class: mask
```

## Docker 

To run docker you''ll need next steps in terminal:
1)Clone docker : 
```bash
docker push 5nevil/image_classification:v2
```
2)Run docker , for example with command :
```bash 
docker run --rm -it -p 8020:12345 5nevil/image_classification:v2
```
3) Open another terminal and make POST curl request :
```bash
curl -X POST -d "urltoimage" http://localhost:8020
```
Wait a seconds and you'll see a predicted class like this :
```bash
Class: mask
```

That all!
